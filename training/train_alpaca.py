"""Alpaca instruction-tuning entry point. Run via
``python -m training.train_alpaca --model_name_or_path <path> ...``."""
import gc
import json
import logging
import math
import os
import time
import datasets
import torch
from accelerate import Accelerator
from datasets import load_dataset
from dotenv import load_dotenv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

import loralib as lora
from data.alpaca import preprocess_dataset
from evaluation.alpaca_generation import evaluate_alpaca_generation
from evaluation.mt_bench import run_mt_bench_evaluation
from modeling.model_setup import prepare_model_for_alpaca
from pruning import pruning_config, pruning_utils
from pruning.stage2.progressive_pruning import ProgressivePruningManager
from training.args_alpaca import parse_args
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


load_dotenv()


logger = logging.getLogger(__name__)


def _smart_tokenizer_and_embedding_resize(tokenizer, model):
    """Resize embeddings; mean-init new-token rows (plausible start vs random).
    fork_rng isolates the mean-compute RNG so it doesn't perturb the training RNG."""
    import torch
    pre_vocab = model.config.vocab_size
    num_new_tokens = len(tokenizer) - pre_vocab
    if num_new_tokens <= 0:
        # Even if no new tokens, still resize (no-op) for symmetry.
        model.resize_token_embeddings(len(tokenizer))
        return

    _rng_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=_rng_devices):
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings_module = model.get_output_embeddings()
        input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_avg
        if output_embeddings_module is not None:
            output_embeddings = output_embeddings_module.weight.data
            output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_avg


def _save_clean_merged_model(unwrapped_model, tokenizer, base_model_name,
                              output_dir, logger):
    """merge_and_unload-equivalent save: clean LoRA-free ``final_model``. Fail-fast on
    any unmerged r>0 layer (``weight`` holds ΔW only after eval-mode merge, else ΔW lost)."""
    unmerged = [
        n for n, m in unwrapped_model.named_modules()
        if hasattr(m, "lora_A") and hasattr(m, "lora_B")
        and getattr(m, "r", 0) > 0 and not getattr(m, "merged", False)
    ]
    if unmerged:
        raise RuntimeError(
            f"_save_clean_merged_model: {len(unmerged)} LoRA layer(s) not merged "
            f"(e.g. {unmerged[:3]}); saving now would silently drop their ΔW. "
            f"Run model.eval()/explicit merge before saving."
        )
    fresh = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=unwrapped_model.dtype
    )
    if len(tokenizer) != fresh.config.vocab_size:
        _smart_tokenizer_and_embedding_resize(tokenizer, fresh)
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
        f"(copied {copied} base keys, LoRA-free)"
    )
    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_best_merged_model(base_model_name, dtype, adapter_state, live_model,
                            tokenizer, output_dir, logger):
    """Materialise ``best_model`` once from an in-RAM adapter snapshot: fresh base +
    ΔW = scaling*(B@A) per layer (in-place add), avoiding per-epoch merge/unmerge drift."""
    fresh = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    if len(tokenizer) != fresh.config.vocab_size:
        _smart_tokenizer_and_embedding_resize(tokenizer, fresh)

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
            if a_key not in adapter_state or b_key not in adapter_state:
                raise RuntimeError(
                    f"best adapter snapshot missing {a_key}/{b_key}"
                )
            delta = adapter_state[b_key].float() @ adapter_state[a_key].float()
            if fan_in_fan_out:
                delta = delta.T
            # bf16 weight += fp32 delta -> fp32 accumulate, round once. Do NOT
            # cast delta to bf16 first. Matches the explicit final merge loop.
            fresh_mods[name].weight.data += delta * scaling
            applied += 1

    fresh.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(
        f"Best merged model saved to {output_dir} "
        f"(applied {applied} LoRA deltas, LoRA-free)"
    )
    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _AlpacaCollator:
    """Dynamic ("longest") padding collator: pad each batch to its longest
    sequence rather than ``max_length`` (more memory-efficient)."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, instances):
        input_ids = [torch.tensor(i["input_ids"], dtype=torch.long) for i in instances]
        labels = [torch.tensor(i["labels"], dtype=torch.long) for i in instances]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_id).long(),
        }


def main():
    args = parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    # Mitigate CUDA memory fragmentation for memory-intensive Alpaca runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.use_memory_efficient_importance:
        os.environ["USE_MEMORY_EFFICIENT_IMPORTANCE"] = "true"
        os.environ["IMPORTANCE_CHUNK_SIZE"] = str(args.importance_chunk_size)
    else:
        os.environ["USE_MEMORY_EFFICIENT_IMPORTANCE"] = "false"

    log_level = getattr(logging, args.log_level.upper())
    global logger
    logger = setup_logger("optimal_lora", os.path.join(args.output_dir, "training.log"), level=log_level)

    # Hugging Face authentication for gated LLaMA / Qwen models.
    if "llama" in args.model_name_or_path.lower() or "qwen" in args.model_name_or_path.lower():
        try:
            from huggingface_hub import login
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face using HF_TOKEN")
            else:
                logger.error("=" * 60)
                logger.error("HF_TOKEN not found in environment!")
                logger.error("Please add HF_TOKEN to your .env file:")
                logger.error("  HF_TOKEN=your_token_here")
                logger.error("Or set environment variable:")
                logger.error("  export HF_TOKEN='your_token'")
                logger.error("=" * 60)
                raise ValueError("Hugging Face token required for LLaMA/Qwen models")
        except Exception as e:
            logger.error(f"Hugging Face authentication failed: {e}")
            raise

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
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=True,
    )

    # Padding token must differ from EOS to avoid corrupting the training signal.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        model_name_lower = args.model_name_or_path.lower()
        if "llama" in model_name_lower:
            if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
                logger.info(f"Set pad_token to Llama finetune pad: id={tokenizer.pad_token_id}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info(f"Added [PAD] for Llama (no finetune_right_pad_id): id={tokenizer.pad_token_id}")
        elif "qwen" in model_name_lower:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Added [PAD] token for Qwen: id={tokenizer.pad_token_id}")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Added [PAD] token (fallback): id={tokenizer.pad_token_id}")

    if args.dataset_path and os.path.exists(args.dataset_path):
        with open(args.dataset_path, 'r') as f:
            data = json.load(f)
        raw_datasets = datasets.Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})
        raw_datasets = datasets.DatasetDict({"train": raw_datasets})
    else:
        # trust_remote_code is safe — this is the official Stanford Alpaca dataset.
        raw_datasets = load_dataset("tatsu-lab/alpaca", trust_remote_code=True)

    if args.eval_data_path and os.path.exists(args.eval_data_path):
        with open(args.eval_data_path, 'r') as f:
            eval_data = json.load(f)
        eval_raw = datasets.Dataset.from_dict(
            {k: [d[k] for d in eval_data] for k in eval_data[0].keys()}
        )
        raw_datasets["validation"] = eval_raw
        logger.info(f"loaded eval data from {args.eval_data_path} ({len(eval_raw)} samples)")

    train_dataset, eval_dataset = preprocess_dataset(
        tokenizer=tokenizer,
        raw_datasets=raw_datasets,
        max_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        logger=logger,
        eval_split_ratio=args.eval_split_ratio,
    )

    data_collator = _AlpacaCollator(tokenizer)

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

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
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

    # Stage 1: ILP rank allocation
    logger.info("Loading base model for Alpaca task...")
    is_llm = ("llama" in args.model_name_or_path.lower()
              or "qwen" in args.model_name_or_path.lower())

    if not is_llm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )

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
        if is_llm:
            logger.info("Loading temporary LLaMA/Qwen model for optimization...")
            temp_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cpu",
            )
            if len(tokenizer) != temp_model.config.vocab_size:
                _smart_tokenizer_and_embedding_resize(tokenizer, temp_model)
            temp_model.config.pad_token_id = tokenizer.pad_token_id
            model = temp_model

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

            if not is_llm:
                total_params = sum(
                    r_value * (get_layer_size(model, layer_name)[0] + get_layer_size(model, layer_name)[1])
                    for layer_name, r_value in r_config.items() if r_value > 0
                )
                logger.info(f"Total LoRA parameters: {total_params:,}")
            else:
                logger.info("Total LoRA parameters: Will be calculated after model loading")
            logger.info("=" * 50)

        if is_llm:
            del temp_model
            del model
            torch.cuda.empty_cache()

        del optimization_dataset, optimization_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Uniform-r path (PEFT canonical): all LoRA target layers get
        # r = args.lora_r (default 8). No ILP, no Stage-1 importance.
        if is_llm:
            logger.info("Loading temporary LLaMA/Qwen model to detect uniform-r target layers...")
            temp_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cpu",
            )
            if len(tokenizer) != temp_model.config.vocab_size:
                _smart_tokenizer_and_embedding_resize(tokenizer, temp_model)
            temp_model.config.pad_token_id = tokenizer.pad_token_id
            model_for_detection = temp_model
        else:
            model_for_detection = model

        opt = LoRAOptimizer(
            model=model_for_detection,
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

        if is_llm:
            del temp_model, model_for_detection
            torch.cuda.empty_cache()

    if args.optimize_only:
        logger.info("Optimization completed. Exiting as --optimize_only was specified.")
        return

    logger.info("Preparing model with LoRA for training...")
    model, replaced_layers = prepare_model_for_alpaca(
        base_model_name=args.model_name_or_path,
        r_config=r_config,
        dropout=args.lora_dropout,
        tokenizer=tokenizer,
    )

    # Resize embeddings to the (possibly extended) tokenizer (Stanford Alpaca mean-init).
    if len(tokenizer) != model.config.vocab_size:
        logger.info(f"Resizing token embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        _smart_tokenizer_and_embedding_resize(tokenizer, model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled")

    validate_lora_configuration(r_config, replaced_layers, logger)

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "model_architecture.txt"), 'w') as f:
            f.write(str(model))

    lora.mark_only_lora_as_trainable(model)
    # No-op for CausalLM (no 'classifier' params); kept for parity with train_glue.
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
    best_adapter_state = None  # in-RAM best LoRA snapshot
    best_epoch = None
    train_start_time = time.time()
    total_train_loss = 0
    total_train_samples = 0
    total_train_steps = 0
    total_micro_batches = 0  # accurate denominator for avg loss
    skipped_micro_batches = 0  # non-finite micro-batches excluded from grad + avg
    window_finite_count = 0  # finite micro-batches in the current accumulation window

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
                    label_tensor = None
                    for key in ('labels', 'label', 'target', 'targets'):
                        if key in batch:
                            label_tensor = batch[key]
                            break
                    if label_tensor is None:
                        raise ValueError(
                            f"Cannot compute loss: no labels in batch. "
                            f"Keys: {list(batch.keys())}. Alpaca preprocessing "
                            f"must produce 'labels' (see data/alpaca.py)."
                        )
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = label_tensor[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                else:
                    raise ValueError(
                        f"Cannot compute loss: no loss or logits in outputs. "
                        f"Output attrs: {dir(outputs)}"
                    )

            if loss is not None:
                if not torch.isfinite(loss):
                    # Non-finite loss: skip backward (keep bad grads out) + skip accumulation
                    # (avg-loss denom stays clean). Cause: empty-label batch (truncated) at bs=1.
                    skipped_micro_batches += 1
                    logger.warning(
                        f"Non-finite training loss at epoch {epoch}, step {step}; "
                        f"skipping micro-batch (no backward, excluded from average)."
                    )
                else:
                    train_loss += loss.detach().float()
                    total_train_loss += loss.detach().float()
                    total_micro_batches += 1
                    window_finite_count += 1

                    batch_size = batch["input_ids"].size(0)
                    total_train_samples += batch_size

                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                # Boundary check uses ``active_train_dataloader`` so a mid-epoch
                # resume still steps on the actual final micro-batch.
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(active_train_dataloader) - 1:
                    if window_finite_count == 0:
                        # Whole window non-finite: flush empty grads, skip optimizer/scheduler
                        # step (no LR/step advance). Single-process; multi-GPU needs all-rank consensus.
                        optimizer.zero_grad()
                        continue
                    window_finite_count = 0

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

        model.eval()
        check_lora_merge_status(model)

        logger.info("Running Alpaca evaluation...")
        eval_start_time = time.time()
        eval_metric = evaluate_alpaca_generation(
            model=model,
            eval_dataloader=eval_dataloader,
            tokenizer=tokenizer,
            accelerator=accelerator,
            max_new_tokens=128,
        )
        eval_runtime = time.time() - eval_start_time
        eval_steps_per_second = (
            len(eval_dataloader) / eval_runtime if eval_runtime > 0 else 0.0
        )
        logger.info(f"Epoch {epoch + 1} evaluation results:")
        logger.info(f"  Perplexity: {eval_metric.get('perplexity', 0):.2f}")
        logger.info(f"  Eval Loss: {eval_metric.get('eval_loss', 0):.4f}")

        eval_metric["eval_steps_per_second"] = eval_steps_per_second
        logger.info(f"Epoch {epoch+1}: {eval_metric}")

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

        # Track best model using inverse perplexity (higher is better).
        current_metric = -eval_metric.get("perplexity", float('inf'))
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            if accelerator.is_main_process:
                # Snapshot best LoRA adapter to CPU RAM. lora_state_dict (A/B) is
                # merge-independent; best_model is materialised once after training.
                unwrapped_model = accelerator.unwrap_model(model)
                # best_model = fresh base + LoRA ΔW only → fail fast if any
                # non-LoRA param was trained (it would be lost in best_model).
                _bad = [n for n, p in unwrapped_model.named_parameters()
                        if p.requires_grad and "lora_" not in n]
                if _bad:
                    raise RuntimeError(
                        f"best_model assumes LoRA-only training; non-LoRA "
                        f"trainable params would be lost: {_bad[:5]}"
                    )
                best_adapter_state = {
                    k: v.detach().cpu().clone()
                    for k, v in lora.lora_state_dict(unwrapped_model).items()
                }
                logger.info(
                    f"New best {args.task_name} metric: {current_metric:.4f} "
                    f"(epoch {best_epoch}); best adapter snapshotted"
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    train_runtime = time.time() - train_start_time

    if skipped_micro_batches > 0:
        logger.warning(
            f"Training finished with {skipped_micro_batches} non-finite "
            f"micro-batch(es) skipped (excluded from gradients and avg loss)."
        )

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

    # Average over the actual micro-batch count; total_train_steps × grad_acc
    # would miss a partial last batch.
    avg_train_loss = (
        (total_train_loss / total_micro_batches).item()
        if total_micro_batches > 0 else 0
    )

    # dtype-aware size: sum element_size (correct under mixed dtype, e.g. bf16
    # base + fp32 LoRA, unlike sampling one parameter's dtype).
    model_size_mb = (
        sum(p.numel() * p.element_size() for p in model.parameters())
        / (1024 * 1024)
    )

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
    unwrapped_model = accelerator.unwrap_model(model)
    logger.info("Ensuring optimal LoRA state for final evaluation...")
    check_lora_merge_status(unwrapped_model)

    eval_start_time = time.time()
    final_eval_results = evaluate_alpaca_generation(
        model=model,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        accelerator=accelerator,
        max_new_tokens=128,
    )
    eval_runtime = time.time() - eval_start_time

    total_eval_samples = final_eval_results.get("num_samples", 0)
    eval_samples_per_second = (
        total_eval_samples / eval_runtime if eval_runtime > 0 else 0.0
    )
    eval_steps_per_second = (
        len(eval_dataloader) / eval_runtime if eval_runtime > 0 else 0.0
    )
    logger.info(f"Final evaluation completed: total runtime={eval_runtime:.2f}s")
    avg_eval_loss = final_eval_results.get("eval_loss", 0)
    formatted_eval_runtime = format_runtime(eval_runtime)
    if total_eval_samples > 0:
        logger.info(f"Total eval samples: {total_eval_samples}")

    if pruning_manager is not None and accelerator.is_main_process:
        # Alpaca has no automatic metric (human eval; AlpacaEval LC-WR needs GPT-4);
        # use inverse perplexity as a sanity check.
        task_performance = 1.0 / final_eval_results.get("perplexity", float('inf'))

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
            "eval_steps_per_second": eval_steps_per_second,
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
        logger.info(f"  Eval steps per second: {eval_steps_per_second:.2f}")
        logger.info(f"  Pruned model size: {pruning_metrics['pruned_model_size_mb']:.2f} MB")
        logger.info(f"  Size reduction: {achieved_reduction*100:.2f}%")
        logger.info(f"  Final performance: {task_performance:.4f}")
        logger.info("=" * 80)

    eval_metric = final_eval_results
    combined_score = 1.0 / eval_metric.get("perplexity", float('inf'))

    eval_metrics = {
        "epoch": args.num_train_epochs,
        "eval_perplexity": eval_metric.get("perplexity", 0),
        "eval_combined_score": combined_score,
        "eval_loss": avg_eval_loss,
        "eval_runtime_formatted": formatted_eval_runtime,
        "eval_samples": total_eval_samples,
        "eval_samples_per_second": eval_samples_per_second,
        "eval_steps_per_second": (
            len(eval_dataloader) / eval_runtime if eval_runtime > 0 else 0
        ),
    }

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

    # merge_weights=False: model.eval() does NOT auto-merge, so ΔW stays in lora_A/B
    # until this single fold loop (`if merged: skip` keeps it idempotent).
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
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
                        # Keep delta fp32: bf16 base += fp32 delta accumulates in
                        # fp32 and rounds once. Casting delta to bf16 first is worse.
                        module.weight.data += delta * module.scaling
                        module.merged = True
                        merged_count += 1
        logger.info(
            f"Merged LoRA weights for {merged_count} layers "
            f"(skipped {skipped_count} already-merged)"
        )

    # Saved before MT-Bench, which reads final_model directly (no tempdir copy).
    if accelerator.is_main_process:
        _save_clean_merged_model(
            unwrapped_model, tokenizer, args.model_name_or_path,
            os.path.join(args.output_dir, "final_model"), logger,
        )
        # No separable adapter written; final_model is the merged full model only
        # (re-applying the adapter would double-apply ΔW).

        # Materialize best_model once from the in-RAM best-adapter snapshot
        # (base loaded a single time). Same on-disk format as final_model.
        if best_adapter_state is not None:
            _save_best_merged_model(
                base_model_name=args.model_name_or_path,
                dtype=unwrapped_model.dtype,
                adapter_state=best_adapter_state,
                live_model=unwrapped_model,
                tokenizer=tokenizer,
                output_dir=os.path.join(args.output_dir, "best_model"),
                logger=logger,
            )
            logger.info(
                f"best_model materialized from epoch {best_epoch} "
                f"(metric {best_metric:.4f})"
            )

    # Now reads the just-saved final_model (LoRA-free vanilla HF). Avoids the
    # multi-GB tempdir copy and the lora_A/lora_B unexpected-keys warning.
    mt_bench_results = None
    if args.run_mt_bench and accelerator.is_main_process:
        logger.info("***** Running MT-Bench evaluation *****")
        logger.info(f"MT-Bench results will be saved to: {args.mt_bench_output_dir}")

        mt_bench_results = run_mt_bench_evaluation(
            model_path=os.path.join(args.output_dir, "final_model"),
            output_dir=args.mt_bench_output_dir,
            logger=logger,
            skip_judgment=args.skip_mt_bench_judgment,
            fastchat_path=args.fastchat_path,
        )

        if mt_bench_results:
            eval_metrics["mt_bench_score"] = mt_bench_results.get("overall_score", -1)
            eval_metrics["mt_bench_judgment_skipped"] = mt_bench_results.get("judgment_skipped", False)
            eval_metrics["mt_bench_answer_file"] = mt_bench_results.get("answer_file", "")

            if "overall_score" in mt_bench_results:
                logger.info(f"MT-Bench Overall Score: {mt_bench_results['overall_score']:.2f}")
            else:
                logger.info("MT-Bench answers generated. Judgment skipped or pending.")
        else:
            logger.warning("MT-Bench evaluation failed or was skipped")
            eval_metrics["mt_bench_score"] = -1
            eval_metrics["mt_bench_error"] = True

    if accelerator.is_main_process:

        save_metrics(eval_metrics, os.path.join(args.output_dir, "eval_metrics.json"))
        save_metrics(
            {**train_metrics, **eval_metrics},
            os.path.join(args.output_dir, "all_metrics.json"),
        )

    logger.info(f"Training completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
