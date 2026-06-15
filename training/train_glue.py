"""GLUE training entry point. Run via
``python -m training.train_glue --task_name <task> ...``."""
import gc
import json
import logging
import math
import os
import time
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)

import loralib as lora
from data.glue import make_compute_metrics, preprocess_dataset
from evaluation.glue import evaluate_glue
from modeling.model_setup import prepare_model_for_glue
from pruning import pruning_config, pruning_utils
from pruning.stage2.progressive_pruning import ProgressivePruningManager
from training.args_glue import parse_args
from utils.determinism import seed_worker, set_deterministic_environment
from utils.common import (
    format_runtime,
    get_layer_size,
    get_memory_stats,
    save_metrics,
    setup_minimal_logging,
    validate_model_parameters,
)
from utils.logging_utils import setup_logger
from pruning.stage1.lora_optimizer import LoRAOptimizer
from pruning.stage1.lora_setup import (
    check_lora_merge_status,
    optimize_lora_config,
    validate_lora_configuration,
)
from utils.save_resume import load_full_state, save_full_state


logger = logging.getLogger(__name__)


def _glue_combined_score(task_name, eval_metric, mismatched_acc=None):
    """GLUE combined_score (higher better) for best-model selection + final score:
    CoLA=Matthews, MRPC/QQP=(acc+f1)/2, MNLI=(matched+mismatched)/2, STS-B=corr, else acc."""
    if task_name == "cola":
        return eval_metric.get("matthews_correlation", 0)
    if task_name in ("mrpc", "qqp"):
        return (eval_metric.get("accuracy", 0) + eval_metric.get("f1", 0)) / 2
    if task_name == "mnli" and mismatched_acc is not None:
        return (eval_metric.get("accuracy", 0) + mismatched_acc) / 2
    if task_name == "stsb":
        return eval_metric.get("corr", 0)
    return eval_metric.get("accuracy", 0)


def _save_clean_merged_model_glue(unwrapped_model, tokenizer, base_model_name,
                                  config, output_dir, logger):
    """final_model save (SequenceClassification): full-copy the eval-merged state_dict so
    a fresh from_pretrained reproduces the run. Fail-fast on any unmerged r>0 layer (ΔW lost)."""
    unmerged = [
        n for n, m in unwrapped_model.named_modules()
        if hasattr(m, "lora_A") and hasattr(m, "lora_B")
        and getattr(m, "r", 0) > 0 and not getattr(m, "merged", False)
    ]
    if unmerged:
        raise RuntimeError(
            f"_save_clean_merged_model_glue: {len(unmerged)} LoRA layer(s) not "
            f"merged (e.g. {unmerged[:3]}); saving now would silently drop ΔW. "
            f"Run model.eval()/explicit merge before saving."
        )
    fresh = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, config=config, torch_dtype=unwrapped_model.dtype
    )
    merged_sd = unwrapped_model.state_dict()
    fresh_sd = fresh.state_dict()
    copied = 0
    for key in fresh_sd.keys():
        if key in merged_sd:
            fresh_sd[key] = merged_sd[key]
            copied += 1
    fresh.load_state_dict(fresh_sd)
    fresh.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(
        f"Clean merged model saved to {output_dir} "
        f"(copied {copied} base+head keys, LoRA-free)"
    )
    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_best_merged_model_glue(base_model_name, config, dtype, best_state,
                                 live_model, tokenizer, output_dir, logger):
    """best_model save: load fresh base once, restore run-specific non-LoRA params from
    the snapshot, apply best ΔW = scaling*(B@A) per Linear (fp32 in-place add)."""
    fresh = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, config=config, torch_dtype=dtype
    )
    fresh_sd = fresh.state_dict()
    # 1) restore non-LoRA head params (classifier/pooler) by key
    restored = 0
    for k, v in best_state.items():
        if "lora_" not in k and k in fresh_sd:
            fresh_sd[k] = v
            restored += 1
    fresh.load_state_dict(fresh_sd)
    # 2) apply LoRA delta onto the fresh base linears
    meta = {
        name: (float(m.scaling), bool(getattr(m, "fan_in_fan_out", False)))
        for name, m in live_model.named_modules()
        if hasattr(m, "lora_A") and hasattr(m, "lora_B") and getattr(m, "r", 0) > 0
    }
    fresh_mods = dict(fresh.named_modules())
    applied = 0
    with torch.no_grad():
        for name, (scaling, fan_in_fan_out) in meta.items():
            a_key, b_key = name + ".lora_A", name + ".lora_B"
            if a_key not in best_state or b_key not in best_state:
                raise RuntimeError(f"best snapshot missing {a_key}/{b_key}")
            delta = best_state[b_key].float() @ best_state[a_key].float()
            if fan_in_fan_out:
                delta = delta.T
            fresh_mods[name].weight.data += delta * scaling
            applied += 1
    fresh.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(
        f"Best merged model saved to {output_dir} "
        f"(restored {restored} head keys, applied {applied} LoRA deltas, LoRA-free)"
    )
    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)

    if args.use_memory_efficient_importance:
        os.environ["USE_MEMORY_EFFICIENT_IMPORTANCE"] = "true"
        os.environ["IMPORTANCE_CHUNK_SIZE"] = str(args.importance_chunk_size)
    else:
        os.environ["USE_MEMORY_EFFICIENT_IMPORTANCE"] = "false"

    log_level = getattr(logging, args.log_level.upper())
    global logger
    logger = setup_logger("optimal_lora", os.path.join(args.output_dir, "training.log"), level=log_level)

    setup_minimal_logging()
    set_deterministic_environment(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=3 if args.task_name == "mnli" else 1 if args.task_name == "stsb" else 2,
        finetuning_task=args.task_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=True,
    )

    # trust_remote_code is safe — GLUE is an official HuggingFace dataset.
    raw_datasets = load_dataset("glue", args.task_name, trust_remote_code=True)

    if args.task_name == "mnli":
        train_dataset, eval_matched_dataset, eval_mismatched_dataset = preprocess_dataset(
            tokenizer=tokenizer,
            raw_datasets=raw_datasets,
            task_name=args.task_name,
            max_length=args.max_seq_length,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
    else:
        train_dataset, eval_dataset = preprocess_dataset(
            tokenizer=tokenizer,
            raw_datasets=raw_datasets,
            task_name=args.task_name,
            max_length=args.max_seq_length,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )

    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False,
    )

    # Stage-2 importance dataloader: fixed-N first samples, no shuffle, eval batch —
    # paper Eq.6 deterministic subset. Not accelerator.prepare'd (all ranks see same N).
    _imp_bs = args.per_device_eval_batch_size
    _imp_n = min(
        pruning_config.IMPORTANCE_NUM_BATCHES * _imp_bs,
        pruning_config.IMPORTANCE_MAX_SAMPLES,
        len(train_dataset),
    )
    importance_dataset = train_dataset.select(range(_imp_n))
    importance_dataloader = DataLoader(
        importance_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=_imp_bs,
    )

    if args.task_name == "mnli":
        eval_dataloader = DataLoader(
            eval_matched_dataset, collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
            worker_init_fn=seed_worker,
            generator=g, persistent_workers=False,
        )
        eval_mismatched_dataloader = DataLoader(
            eval_mismatched_dataset, collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
            worker_init_fn=seed_worker,
            generator=g, persistent_workers=False,
        )
    else:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
            worker_init_fn=seed_worker,
            generator=g, persistent_workers=False,
        )

    logger.info("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    # Stage 1: LoRA rank allocation (3-way: existing / ILP / uniform-r). Resume override:
    # build the model with the saved post-pruning r_config so LoRA shapes match the ckpt.
    resume_pruning_state_path = (
        os.path.join(args.resume_from_checkpoint, "pruning_state.json")
        if args.resume_from_checkpoint else None
    )
    if resume_pruning_state_path is not None and os.path.exists(resume_pruning_state_path):
        with open(resume_pruning_state_path, "r") as f:
            saved_pruning_state = json.load(f)
        r_config = {k: int(v) for k, v in saved_pruning_state["current_r_config"].items()}
        logger.info(
            f"resume: loaded saved r_config "
            f"(post-pruning shapes) from {resume_pruning_state_path}"
        )
    elif args.use_existing_lora_config:
        logger.info(f"Using existing LoRA configuration from {args.use_existing_lora_config}")
        with open(args.use_existing_lora_config, 'r') as f:
            r_config = json.load(f)
    elif args.use_initial_rank_allocation:
        _opt_bs = args.per_device_eval_batch_size
        _opt_n = min(
            pruning_config.IMPORTANCE_NUM_BATCHES * _opt_bs,
            pruning_config.IMPORTANCE_MAX_SAMPLES,
            len(train_dataset),
        )
        optimization_dataset = train_dataset.select(range(_opt_n))
        optimization_dataloader = DataLoader(
            optimization_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=_opt_bs,
        )

        r_config = optimize_lora_config(
            model=model,
            dataloader=optimization_dataloader,
            args=args,
            accelerator=accelerator,
            output_dir=args.output_dir,
        )

        if accelerator.is_main_process:
            logger.info("=" * 50)
            logger.info("OPTIMAL LoRA RANK (r) CONFIGURATION:")
            logger.info("=" * 50)
            for layer_name, r_value in sorted(r_config.items()):
                logger.info(f"  {layer_name}: r = {r_value}")
            if r_config:
                avg_r = sum(r_config.values()) / len(r_config)
                logger.info(f"Average r across all layers: {avg_r:.2f}")
            logger.info("=" * 50)

            total_params = sum(
                r_value * (get_layer_size(model, layer_name)[0] + get_layer_size(model, layer_name)[1])
                for layer_name, r_value in r_config.items() if r_value > 0
            )
            logger.info(f"Total LoRA parameters: {total_params:,}")
            logger.info("=" * 50)

        del optimization_dataset, optimization_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Uniform-r path (PEFT canonical): all LoRA target layers get
        # r = args.lora_r (default 8). No ILP, no Stage-1 importance.
        opt = LoRAOptimizer(
            model=model,
            r_values=[args.lora_r],
            budget=1.0,
            device=accelerator.device,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        target_layers = opt.find_target_layers()
        r_config = {layer: args.lora_r for layer in target_layers}
        if accelerator.is_main_process:
            logger.info(
                f"uniform-r: r={args.lora_r} across {len(r_config)} target layers."
            )

    if args.optimize_only:
        logger.info("Optimization completed. Exiting as --optimize_only was specified.")
        return

    logger.info("Preparing model with LoRA for training...")
    model, replaced_layers = prepare_model_for_glue(
        base_model_name=args.model_name_or_path,
        r_config=r_config,
        num_labels=config.num_labels,
        dropout=args.lora_dropout,
    )

    validate_lora_configuration(r_config, replaced_layers, logger)

    # HF PEFT canonical (modules_to_save): classifier / pooler must NOT be
    # LoRA-wrapped. Fail fast on misconfigured external JSON.
    _disallowed = [k for k in r_config if 'classifier' in k or 'pooler' in k]
    if _disallowed:
        raise ValueError(
            f"classifier/pooler modules cannot be LoRA-wrapped (HF PEFT canonical). "
            f"Found in r_config: {_disallowed}. Remove from --use_existing_lora_config "
            f"or rely on auto-detect, then use --train_classifier_head to full-tune them."
        )

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "model_architecture.txt"), 'w') as f:
            f.write(str(model))

    lora.mark_only_lora_as_trainable(model)
    if args.train_classifier_head:
        for n, p in model.named_parameters():
            if 'classifier' in n:
                p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

    total_lora_params = 0
    for layer_name, r_value in r_config.items():
        if r_value > 0:
            in_features, out_features = get_layer_size(model, layer_name)
            total_lora_params += r_value * (in_features + out_features)
    logger.info(f"Total LoRA parameters: {total_lora_params:,}")

    # AdamW weight-decay group split (HF Trainer canonical). LoRA-only runs leave
    # the no_decay group empty (no `bias`/`LayerNorm.weight`), so behavior is unchanged.
    no_decay = ("bias", "LayerNorm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        no_deprecation_warning=True,
    )

    if args.task_name == "mnli":
        model, optimizer, train_dataloader, eval_dataloader, eval_mismatched_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, eval_mismatched_dataloader
        )
    else:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        if num_update_steps_per_epoch > 0:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        else:
            logger.error(f"Training dataloader is too small: {len(train_dataloader)} batches")
            raise ValueError(
                f"Not enough training data: {len(train_dataloader)} batches with "
                f"gradient_accumulation_steps={args.gradient_accumulation_steps}"
            )

    if args.num_warmup_steps == 0 and args.warmup_ratio > 0:
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    metric_fn = make_compute_metrics(args.task_name)

    # Stage 2: progressive pruning manager
    pruning_manager = None
    if args.apply_pruning:
        logger.info("=" * 80)
        logger.info("INITIALIZING PROGRESSIVE LORA PRUNING")
        logger.info("=" * 80)

        unwrapped_model = accelerator.unwrap_model(model)
        _resume_pp = (
            os.path.join(args.resume_from_checkpoint, "pruning_state.json")
            if args.resume_from_checkpoint else None
        )
        if _resume_pp is not None and os.path.exists(_resume_pp):
            # AdaLoRA-standard: anchor the budget schedule to the ORIGINAL b^0 (not the
            # pruned state). b^0 = pruning_history[0] (status='initial'), persisted in JSON.
            with open(_resume_pp, "r") as f:
                _ps = json.load(f)
            _hist = _ps.get("pruning_history") or []
            if _hist and _hist[0].get("status") == "initial":
                initial_r_config = {k: int(v) for k, v in _hist[0]["r_config"].items()}
                logger.info(
                    "resume: pruning baseline restored from pruning_history[0] "
                    "(original initial_r_config / budget)"
                )
            else:
                initial_r_config = {k: int(v) for k, v in _ps["current_r_config"].items()}
                logger.warning(
                    "resume: pruning_history[0] missing 'initial' snapshot; "
                    "baseline falls back to current_r_config (budget may be off)"
                )
        elif not hasattr(unwrapped_model, 'initial_r_config'):
            logger.warning("Model doesn't have initial_r_config attribute. Extracting from model...")
            initial_r_config = {}
            for name, module in unwrapped_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    r = getattr(module, 'lora_A').shape[0] if hasattr(module, 'lora_A') else 0
                    if r > 0:
                        initial_r_config[name] = r
        else:
            initial_r_config = unwrapped_model.initial_r_config

        pruning_output_dir = args.pruning_output_dir or os.path.join(args.output_dir, "pruning")
        os.makedirs(pruning_output_dir, exist_ok=True)

        # Pass pruning hyperparameters via kwargs instead of mutating
        # ``pruning_config`` module globals.
        pruning_manager = ProgressivePruningManager(
            model=unwrapped_model,
            initial_r_config=initial_r_config,
            train_dataloader=train_dataloader,
            importance_dataloader=importance_dataloader,
            eval_dataloader=eval_dataloader,
            target_reduction=args.pruning_target_reduction,
            num_pruning_steps=args.pruning_steps,
            total_training_steps=args.max_train_steps,
            device=accelerator.device,
            output_dir=pruning_output_dir,
            seed=args.seed,
            warmup_steps=args.num_warmup_steps,
            enable_rollback=args.enable_rollback,
            importance_ema_decay=args.importance_ema_decay,
            momentum_penalty_weight=args.momentum_penalty_weight,
            stable_layer_bonus=args.stable_layer_bonus,
            recovery_steps=args.recovery_steps,
            extended_recovery_steps=args.extended_recovery_steps,
            available_r_values=args.lora_r_values,
        )

        pruning_manager.initialize_baseline_performance()
        check_lora_merge_status(model)

    logger.info(f"Starting training for {args.num_train_epochs} epochs ({args.max_train_steps} steps)")
    logger.info(
        f"Training parameters: batch_size={args.per_device_train_batch_size}, "
        f"lr={args.learning_rate}, grad_accum={args.gradient_accumulation_steps}, "
        f"max_grad_norm={args.max_grad_norm}"
    )
    completed_steps = 0
    start_epoch = 0
    steps_trained_in_current_epoch = 0
    live_generator_state = None  # live ``g`` at checkpoint (mid-epoch resume resync)
    best_metric = -float('inf')
    best_state = None  # in-RAM best snapshot (adapter + head)
    best_epoch = None
    train_start_time = time.time()
    total_train_loss = 0
    total_train_samples = 0
    total_train_steps = 0
    total_micro_batches = 0  # avg-loss denominator (micro-batch unit)

    # Resume from checkpoint (no-op when None).
    if args.resume_from_checkpoint is not None:
        unwrapped_for_resume = accelerator.unwrap_model(model)
        resume_state = load_full_state(
            args.resume_from_checkpoint, unwrapped_for_resume,
            optimizer, lr_scheduler, pruning_manager,
            dataloader_generator=g,
        )
        completed_steps = resume_state["completed_steps"]
        start_epoch = resume_state["start_epoch"]
        steps_trained_in_current_epoch = resume_state.get(
            "steps_trained_in_current_epoch", 0
        )
        live_generator_state = resume_state.get("dataloader_generator_live_state")
        logger.info(
            f"Resumed from {args.resume_from_checkpoint}: "
            f"completed_steps={completed_steps}, start_epoch={start_epoch}, "
            f"steps_trained_in_current_epoch={steps_trained_in_current_epoch}"
        )

    if not validate_model_parameters(model):
        raise RuntimeError(
            "Model parameter sanity check failed before training (see logs above)."
        )

    for epoch in range(start_epoch, args.num_train_epochs):
        # Outer guard: ensure max_train_steps is honored even when it does not
        # align with epoch boundaries.
        if completed_steps >= args.max_train_steps:
            break

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            logger.info(
                f"Epoch {epoch+1}/{args.num_train_epochs} - Starting memory: "
                f"{torch.cuda.memory_allocated() / (1024**3):.2f}GB"
            )

        model.train()
        train_loss = 0.0

        # Snapshot the dataloader generator at the START of the epoch (before its
        # shuffle perm is drawn) so a mid-epoch resume re-draws the SAME perm.
        epoch_start_generator_state = g.get_state()

        # Mid-epoch resume: skip the first N micro-batches so the loop restarts
        # where the checkpoint stopped (N = steps_trained_in_current_epoch × grad_acc).
        if epoch == start_epoch and steps_trained_in_current_epoch > 0:
            from accelerate import skip_first_batches
            n_skip_micro = (
                steps_trained_in_current_epoch * args.gradient_accumulation_steps
            )
            active_train_dataloader = skip_first_batches(
                train_dataloader, n_skip_micro
            )
            logger.info(
                f"mid-epoch resume: skipping first {n_skip_micro} micro-batches "
                f"({steps_trained_in_current_epoch} optimizer steps) in epoch {epoch}"
            )
            # Only the resumed epoch skips; later epochs use the full dataloader.
            steps_trained_in_current_epoch = 0
            # After the resumed epoch re-draws its perm, set ``g`` to its live checkpoint
            # state so earlier-in-epoch consumers reproduce and later epochs stay bit-exact.
            resync_g_to_live = live_generator_state is not None
        else:
            active_train_dataloader = train_dataloader
            resync_g_to_live = False

        for step, batch in enumerate(active_train_dataloader):
            if resync_g_to_live and step == 0:
                g.set_state(live_generator_state)
                resync_g_to_live = False
            if pruning_manager is not None and pruning_manager.should_prune(completed_steps):
                pruning_success = pruning_manager.execute_pruning_step(completed_steps, optimizer=optimizer)
                if pruning_success:
                    unwrapped_model = accelerator.unwrap_model(model)
                    logger.info("Checking LoRA layer states after pruning...")
                    check_lora_merge_status(unwrapped_model)

            outputs = model(**batch)

            loss = outputs.loss
            if loss is None:
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    if 'labels' in batch:
                        if batch['labels'].dim() == 1:
                            loss = torch.nn.functional.cross_entropy(outputs.logits, batch['labels'])
                        else:
                            # squeeze(-1): per-batch path, avoid 0-dim scalar when B=1.
                            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(-1), batch['labels'])
                    else:
                        label_found = False
                        for key in ('label', 'target', 'targets'):
                            if key in batch:
                                if batch[key].dim() == 1:
                                    loss = torch.nn.functional.cross_entropy(outputs.logits, batch[key])
                                else:
                                    loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(-1), batch[key])
                                label_found = True
                                break
                        if not label_found:
                            logger.error(
                                f"No labels found in batch. Skipping this batch. "
                                f"Batch keys: {list(batch.keys())}"
                            )
                            continue
                else:
                    logger.warning(
                        f"Cannot compute loss: No loss or logits in outputs. "
                        f"Output keys: {dir(outputs)}"
                    )
                    continue

            if loss is not None:
                train_loss += loss.detach().float()
                # HF Trainer pattern: device-resident tensor accumulator, .item()
                # only at logging/final sync. Avoids per-step CUDA→CPU sync overhead.
                total_train_loss += loss.detach().float()
                total_micro_batches += 1

                batch_size = batch["input_ids"].size(0)
                total_train_samples += batch_size

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(active_train_dataloader) - 1:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if pruning_manager is not None and pruning_manager.scheduler.in_recovery_mode:
                        pruning_manager.update_recovery(optimizer=optimizer)

                    completed_steps += 1
                    total_train_steps += 1

                    if (args.save_strategy == "steps" and args.save_steps > 0
                            and completed_steps % args.save_steps == 0):
                        save_full_state(
                            os.path.join(args.output_dir, f"checkpoint-{completed_steps}"),
                            accelerator.unwrap_model(model),
                            optimizer, lr_scheduler,
                            completed_steps, epoch, pruning_manager,
                            num_update_steps_per_epoch=num_update_steps_per_epoch,
                            dataloader_generator_state=epoch_start_generator_state,
                            dataloader_generator_live_state=g.get_state(),
                        )

                    if completed_steps % 100 == 0:
                        avg_loss_per_batch = train_loss / (step + 1)
                        avg_loss_per_step  = train_loss / completed_steps if completed_steps > 0 else 0
                        progress = (completed_steps / args.max_train_steps) * 100
                        current_lr = optimizer.param_groups[0]['lr']

                        if pruning_manager is not None:
                            pruning_info = pruning_manager.get_progress_info()
                            pruning_status = f", Pruning: {pruning_info['reduction']:.2%} reduction"
                            if pruning_manager.scheduler.in_recovery_mode:
                                pruning_status += " (in recovery)"
                        else:
                            pruning_status = ""

                        logger.info(
                            f"Step {completed_steps}/{args.max_train_steps} ({progress:.1f}%): "
                            f"Loss/batch={avg_loss_per_batch:.4f}, Loss/step={avg_loss_per_step:.4f}, "
                            f"LR={current_lr:.2e}{pruning_status}"
                        )

                    if completed_steps >= args.max_train_steps:
                        break

        check_lora_merge_status(model)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()

        eval_result = evaluate_glue(
            model=model,
            eval_dataloader=eval_dataloader,
            compute_metrics_fn=metric_fn,
            task_name=args.task_name,
            accelerator=accelerator,
        )
        eval_metric = eval_result["eval_metric"]
        logger.info(f"Epoch {epoch + 1}: {eval_metric}")

        if args.task_name == "mnli":
            mismatched_metric_fn = make_compute_metrics(args.task_name)
            unwrapped_model.eval()
            mm_result = evaluate_glue(
                model=model,
                eval_dataloader=eval_mismatched_dataloader,
                compute_metrics_fn=mismatched_metric_fn,
                task_name=args.task_name,
                accelerator=accelerator,
            )
            mismatched_results = mm_result["eval_metric"]
            logger.info(f"Epoch {epoch + 1} MNLI mismatched: {mismatched_results}")
            eval_metric["accuracy_mismatched"] = mismatched_results.get("accuracy", 0.0)

        eval_metric["eval_steps_per_second"] = eval_result["eval_steps_per_second"]
        eval_metric["eval_loss"] = eval_result["eval_loss"]

        logger.info(f"Epoch {epoch+1}: {eval_metric}")

        # Track best by combined_score; snapshot adapter+head to RAM, materialise
        # best_model once after training (lora A/B merge-independent).
        current_metric = _glue_combined_score(
            args.task_name, eval_metric, eval_metric.get("accuracy_mismatched")
        )
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            if accelerator.is_main_process:
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in lora.lora_state_dict(unwrapped_model).items()
                }
                best_state.update({
                    n: p.detach().cpu().clone()
                    for n, p in unwrapped_model.named_parameters()
                    if "lora_" not in n
                    and (p.requires_grad or "classifier" in n or "pooler" in n)
                })
                logger.info(
                    f"New best {args.task_name} combined_score: {current_metric:.4f} "
                    f"(epoch {best_epoch}); best adapter+head snapshotted"
                )

        if args.save_strategy == "epoch":
            save_full_state(
                os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}"),
                accelerator.unwrap_model(model),
                optimizer, lr_scheduler,
                completed_steps, epoch, pruning_manager,
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                dataloader_generator_state=g.get_state(),
                dataloader_generator_live_state=g.get_state(),
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    train_runtime = time.time() - train_start_time
    # Throughput — both metrics use the same unit
    # (optimizer-step × effective batch).
    effective_batch_size = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes
    )
    train_samples_per_second = (
        (total_train_steps * effective_batch_size) / train_runtime
        if train_runtime > 0 else 0
    )
    train_steps_per_second = (
        total_train_steps / train_runtime if train_runtime > 0 else 0
    )
    # device-resident tensor → sync to CPU once. Average over the actual
    # micro-batch count; total_train_steps × grad_acc would miss a partial last batch.
    avg_train_loss = (
        (total_train_loss / total_micro_batches).item()
        if total_micro_batches > 0 else 0
    )

    model_size_mb = total_params * 4 / (1024 * 1024)
    formatted_train_runtime = format_runtime(train_runtime)

    pruning_metrics = {}
    if pruning_manager is not None:
        pruning_info = pruning_manager.get_progress_info()
        pruning_metrics = pruning_info.get("pruning_metrics", {})

    train_metrics = {
        "epoch": args.num_train_epochs,
        "train_loss": avg_train_loss,
        "train_runtime_formatted": formatted_train_runtime,
        "train_samples": total_train_samples,
        "train_samples_per_second": train_samples_per_second,
        "train_steps_per_second": train_steps_per_second,
        "trainable_lora_parameters": trainable_params,
        "total_parameters": total_params,
        "model_size_mb": model_size_mb,
    }

    if pruning_metrics:
        if pruning_manager is not None:
            current_params = pruning_utils.calculate_total_parameters(
                pruning_manager.current_r_config, pruning_manager.layer_sizes,
            )
            actual_pruned_params = pruning_manager.initial_params - current_params
            actual_reduction_ratio = (
                (pruning_manager.initial_params - current_params) / pruning_manager.initial_params
            )
            pruning_metrics["pruned_params"] = actual_pruned_params
            pruning_metrics["model_size_reduction"] = actual_reduction_ratio * 100
            logger.debug(
                f"Updated pruning metrics: pruned_params={actual_pruned_params:,}, "
                f"reduction={actual_reduction_ratio:.4f} ({actual_reduction_ratio*100:.2f}%)"
            )

        train_metrics.update({
            "pruned_params": pruning_metrics.get("pruned_params", 0),
            "pruned_model_size_mb": model_size_mb * (1 - pruning_metrics.get("model_size_reduction", 0) / 100),
            "model_size_reduction_pct": pruning_metrics.get("model_size_reduction", 0),
        })

    if torch.cuda.is_available():
        memory_stats = get_memory_stats()
        train_metrics.update({
            "peak_memory_gb": memory_stats.get("peak_allocated_gb", 0),
            "memory_reserved_gb": memory_stats.get("reserved_gb", 0),
        })

    logger.info("***** Train metrics *****")
    for key, value in train_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:<25} = {value:>10.6f}")
        else:
            logger.info(f"  {key:<25} = {value:>10}")

    if accelerator.is_main_process:
        save_metrics(train_metrics, os.path.join(args.output_dir, "train_metrics.json"))

    if pruning_manager is not None:
        logger.info("=" * 80)
        logger.info("FINALIZING PROGRESSIVE LORA PRUNING")
        logger.info("=" * 80)
        # finalize() returns the model itself (it is modified in place);
        # we only need the new r_config and reduction ratio here.
        _, final_r_config, achieved_reduction = pruning_manager.finalize()

    logger.info("***** Running final evaluation *****")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model.eval()
    # sklearn/scipy/numpy closure (bit-exact equivalent of HF evaluate "glue").
    metric_fn = make_compute_metrics(args.task_name)

    unwrapped_model = accelerator.unwrap_model(model)
    logger.info("Ensuring optimal LoRA state for final evaluation...")
    check_lora_merge_status(unwrapped_model)
    unwrapped_model.eval()

    final_result = evaluate_glue(
        model=model,
        eval_dataloader=eval_dataloader,
        compute_metrics_fn=metric_fn,
        task_name=args.task_name,
        accelerator=accelerator,
    )
    eval_metric = final_result["eval_metric"]
    avg_eval_loss = final_result["eval_loss"]
    eval_runtime = final_result["eval_runtime"]
    eval_samples_per_second = final_result["eval_samples_per_second"]
    total_eval_samples = final_result["total_samples"]

    formatted_eval_runtime = format_runtime(eval_runtime)
    if total_eval_samples > 0:
        logger.info(f"Total eval samples: {total_eval_samples}")

    mismatched_accuracy = None
    if args.task_name == "mnli":
        mismatched_metric_fn = make_compute_metrics(args.task_name)
        mm_final = evaluate_glue(
            model=model,
            eval_dataloader=eval_mismatched_dataloader,
            compute_metrics_fn=mismatched_metric_fn,
            task_name=args.task_name,
            accelerator=accelerator,
        )
        mismatched_results = mm_final["eval_metric"]
        mismatched_accuracy = mismatched_results.get("accuracy", 0.0)
        total_eval_samples += mm_final["total_samples"]

    combined_score = _glue_combined_score(args.task_name, eval_metric, mismatched_accuracy)

    accuracy_key = "eval_accuracy_matched" if args.task_name == "mnli" else "eval_accuracy"

    eval_metrics = {
        "epoch": args.num_train_epochs,
        accuracy_key: eval_metric.get("accuracy", eval_metric.get("matthews_correlation", 0)),
        "eval_combined_score": combined_score,
        "eval_loss": avg_eval_loss,
        "eval_runtime_formatted": formatted_eval_runtime,
        "eval_samples": total_eval_samples,
        "eval_samples_per_second": eval_samples_per_second,
        "eval_steps_per_second": final_result["eval_steps_per_second"],
    }

    if "f1" in eval_metric:
        eval_metrics["eval_f1"] = eval_metric["f1"]
    if "matthews_correlation" in eval_metric:
        eval_metrics["eval_matthews_correlation"] = eval_metric["matthews_correlation"]
    if "pearson" in eval_metric:
        eval_metrics["eval_pearson"] = eval_metric["pearson"]
    if "spearmanr" in eval_metric:
        eval_metrics["eval_spearmanr"] = eval_metric["spearmanr"]
    if args.task_name == "mnli" and mismatched_accuracy is not None:
        eval_metrics["eval_accuracy_mismatched"] = mismatched_accuracy

    if torch.cuda.is_available():
        memory_stats = get_memory_stats()
        eval_metrics.update({
            "eval_peak_memory_gb": memory_stats.get("peak_allocated_gb", 0),
            "eval_memory_reserved_gb": memory_stats.get("reserved_gb", 0),
        })

    logger.info("***** Eval metrics *****")
    for key, value in eval_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:<25} = {value:>10.6f}")
        else:
            logger.info(f"  {key:<25} = {value:>10}")

    if pruning_manager is not None and accelerator.is_main_process:
        if args.task_name == "cola":
            task_performance = eval_metric.get("matthews_correlation", 0)
        elif args.task_name == "stsb":
            task_performance = eval_metric.get("pearson", 0)
        elif args.task_name == "mnli":
            matched_acc = eval_metric.get("accuracy", 0)
            mismatched_acc = mismatched_accuracy if mismatched_accuracy is not None else 0
            task_performance = (matched_acc + mismatched_acc) / 2
        else:
            task_performance = eval_metric.get("accuracy", 0)

        pruning_metrics = {
            "initial_parameters": pruning_manager.initial_params,
            "final_parameters": pruning_utils.calculate_total_parameters(
                final_r_config, pruning_manager.layer_sizes,
            ),
            "reduction_rate": float(achieved_reduction),
            "target_reduction_rate": args.pruning_target_reduction,
            "pruning_steps": args.pruning_steps,
            "baseline_performance": float(pruning_manager.baseline_performance),
            "final_performance": task_performance,
            "eval_steps_per_second": final_result["eval_steps_per_second"],
            "pruned_model_size_mb": model_size_mb * (1 - achieved_reduction),
            "model_size_reduction_pct": achieved_reduction * 100,
        }

        save_metrics(
            pruning_metrics,
            os.path.join(pruning_manager.output_dir, "pruning_metrics.json"),
        )

        logger.info("=" * 80)
        logger.info("PRUNING EFFICIENCY METRICS (FINAL EVALUATION):")
        logger.info("=" * 80)
        logger.info(f"  Eval steps per second: {final_result['eval_steps_per_second']:.2f}")
        logger.info(f"  Pruned model size: {pruning_metrics['pruned_model_size_mb']:.2f} MB")
        logger.info(f"  Size reduction: {achieved_reduction*100:.2f}%")
        logger.info(f"  Final performance: {task_performance:.4f}")
        logger.info("=" * 80)

    if accelerator.is_main_process:
        save_metrics(eval_metrics, os.path.join(args.output_dir, "eval_metrics.json"))
        save_metrics(
            {**train_metrics, **eval_metrics},
            os.path.join(args.output_dir, "all_metrics.json"),
        )

        # Deployable artifacts (HF standard, LoRA-free): final_model = full-copy of
        # the live merged model; best_model = materialise the in-RAM snapshot once.
        unwrapped_model = accelerator.unwrap_model(model)
        # merge_weights=False: eval() left ΔW unmerged, so fold it into `weight`
        # before the LoRA-free export. No-op when already merged.
        merged_count = 0
        skipped_count = 0
        with torch.no_grad():
            for name, module in unwrapped_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'r'):
                    if module.r > 0:
                        if getattr(module, 'merged', False):
                            skipped_count += 1
                            continue
                        fan_in_fan_out = getattr(module, 'fan_in_fan_out', False)
                        delta = module.lora_B @ module.lora_A
                        if fan_in_fan_out:
                            delta = delta.T
                        module.weight.data += delta * module.scaling
                        module.merged = True
                        merged_count += 1
        logger.info(
            f"Merged LoRA weights for {merged_count} layers "
            f"(skipped {skipped_count} already-merged)"
        )
        _save_clean_merged_model_glue(
            unwrapped_model, tokenizer, args.model_name_or_path, config,
            os.path.join(args.output_dir, "final_model"), logger,
        )
        if best_state is not None:
            _save_best_merged_model_glue(
                args.model_name_or_path, config, unwrapped_model.dtype,
                best_state, unwrapped_model, tokenizer,
                os.path.join(args.output_dir, "best_model"), logger,
            )
            logger.info(
                f"best_model materialized from epoch {best_epoch} "
                f"(combined_score {best_metric:.4f})"
            )

    logger.info(f"Training completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
