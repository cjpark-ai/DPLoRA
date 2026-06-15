"""Argparse definition for the Alpaca training entry point."""
import argparse
import os
import time

from transformers import SchedulerType


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM models with optimal LoRA for Alpaca instruction-following task")

    # Task and model arguments
    parser.add_argument(
        "--task_name",
        type=str,
        default="alpaca",
        help="Task name (default: alpaca)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length after tokenization",
    )

    # LoRA optimization arguments
    parser.add_argument(
        "--lora_r_values",
        type=str,
        default="0,1,2,4,8,16,32",
        help="Comma-separated list of r values to consider for LoRA optimization",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--lora_budget",
        type=float,
        default=400000000.0,
        help="Budget constraint for LoRA optimization",
    )
    # Rank-allocation source (mutually exclusive): --use_existing_lora_config or
    # --use_initial_rank_allocation; if neither, uniform r=args.lora_r.
    rank_alloc_group = parser.add_mutually_exclusive_group()
    rank_alloc_group.add_argument(
        "--use_existing_lora_config",
        type=str,
        default=None,
        help="Path to existing LoRA configuration file (skips optimization phase)",
    )
    # Checkpoint save / resume (HF Trainer-style API).
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["no", "epoch", "steps"],
        default="no",
        help="Checkpoint save strategy. default 'no' (training-result-preserving).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N optimizer steps when --save_strategy=steps.",
    )
    rank_alloc_group.add_argument(
        "--use_initial_rank_allocation",
        action="store_true",
        help="Run the Stage-1 ILP rank allocation. If not specified "
             "(and --use_existing_lora_config is also unset), all target "
             "layers use a uniform rank --lora_r (PEFT canonical default).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Uniform LoRA rank used when neither "
             "--use_existing_lora_config nor --use_initial_rank_allocation "
             "is supplied (PEFT canonical default = 8).",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model, logs, and results",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU/TPU for evaluation (Stanford Alpaca canonical: 4)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after warmup period)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to apply",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping (0 to disable)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps (overrides num_train_epochs)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps for gradient accumulation before optimization "
             "(Stanford Alpaca canonical: 8 → effective batch 128 with 4 GPUs × bs 4)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",  # Stanford Alpaca convention
        help="LR scheduler type (Stanford default: cosine)",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the LR scheduler",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Ratio of total training steps used for warmup (Stanford Alpaca canonical: 0.03)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging: limit the number of training examples",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging: limit the number of evaluation examples",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to local Alpaca dataset JSON file",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="optional path to a separate eval JSON file. If supplied, "
             "it overrides any in-dataset validation split and any --eval_split_ratio.",
    )
    parser.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.0,
        help="when --eval_data_path is not set and the raw dataset has "
             "no validation split, carve out this fraction of train as eval. "
             "0.0 (default) = no eval (Stanford Alpaca canonical).",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--optimize_only",
        action="store_true",
        help="Only perform LoRA optimization without training",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level",
    )

    # Pruning arguments
    parser.add_argument(
        "--apply_pruning",
        action="store_true",
        help="Apply progressive pruning during training",
    )
    parser.add_argument(
        "--pruning_target_reduction",
        type=float,
        default=0.5,
        help="Target parameter reduction for progressive pruning (0.0-1.0)",
    )
    parser.add_argument(
        "--pruning_steps",
        type=int,
        default=4,
        help="Number of pruning steps for progressive pruning",
    )
    parser.add_argument(
        "--pruning_output_dir",
        type=str,
        default=None,
        help="Directory to save pruning results (defaults to output_dir/pruning)",
    )
    parser.add_argument(
        "--importance_ema_decay",
        type=float,
        default=0.6,
        help="EMA decay factor for layer importance scores (0.0-1.0)",
    )
    parser.add_argument(
        "--momentum_penalty_weight",
        type=float,
        default=0.25,
        help="Weight for momentum-based penalty in pruning",
    )
    parser.add_argument(
        "--stable_layer_bonus",
        type=float,
        default=0.05,
        help="Bonus for keeping r values stable across pruning steps (delta)",
    )

    parser.add_argument(
        "--recovery_steps",
        type=int,
        default=500,
        help="Number of recovery steps after pruning",
    )
    parser.add_argument(
        "--extended_recovery_steps",
        type=int,
        default=1000,
        help="Number of extended recovery steps after rollback",
    )

    parser.add_argument(
        "--enable_rollback",
        action="store_true",
        default=False,
        help="Enable pruning rollback when performance drops more than threshold "
             "(default: False — rollback disabled; paper does not use rollback).",
    )

    # Memory efficiency arguments
    parser.add_argument(
        "--use_memory_efficient_importance",
        action="store_true",
        help="Use memory-efficient chunking for importance calculation (reduces VRAM usage)",
    )
    parser.add_argument(
        "--importance_chunk_size",
        type=int,
        default=8,
        help="Chunk size for memory-efficient importance calculation (default: 8)",
    )

    # MT-Bench evaluation arguments
    parser.add_argument(
        "--run_mt_bench",
        action="store_true",
        help="Run MT-Bench evaluation after final training",
    )
    parser.add_argument(
        "--mt_bench_output_dir",
        type=str,
        default=None,
        help="Directory to save MT-Bench results (default: output_dir/mt_bench)",
    )
    parser.add_argument(
        "--skip_mt_bench_judgment",
        action="store_true",
        help="Skip GPT-4 judgment (useful when OpenAI API key not available)",
    )
    parser.add_argument(
        "--fastchat_path",
        type=str,
        default=None,
        help="Path to FastChat repository (will auto-detect if not provided)",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )

    parser.add_argument(
        "--train_classifier_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the classifier head together with LoRA params. "
             "No-op for CausalLM models; provided for symmetry with run_glue.py.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "runs",
            f"{args.task_name}_{args.model_name_or_path.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}",
        )

    raw_lr_values = args.lora_r_values
    parsed_r = []
    for tok in raw_lr_values.split(","):
        s = tok.strip()
        if not s:
            raise ValueError(
                f"--lora_r_values has empty token in '{raw_lr_values}'; "
                f"use comma-separated non-negative integers (e.g. '0,1,2,4,8')."
            )
        try:
            v = int(s)
        except ValueError:
            raise ValueError(
                f"--lora_r_values has non-integer token '{s}' in '{raw_lr_values}'."
            )
        if v < 0:
            raise ValueError(
                f"--lora_r_values has negative value {v} in '{raw_lr_values}'; "
                f"all values must be >= 0 (0 lets ILP drop a layer)."
            )
        parsed_r.append(v)
    args.lora_r_values = parsed_r

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"--gradient_accumulation_steps must be >= 1, got {args.gradient_accumulation_steps}"
        )
    if args.lora_budget <= 0:
        raise ValueError(f"--lora_budget must be > 0, got {args.lora_budget}")
    if args.importance_chunk_size < 1:
        raise ValueError(
            f"--importance_chunk_size must be >= 1, got {args.importance_chunk_size}"
        )
    if args.lora_r < 1:
        raise ValueError(f"--lora_r must be >= 1, got {args.lora_r}")
    if args.max_seq_length < 1:
        raise ValueError(
            f"--max_seq_length must be >= 1, got {args.max_seq_length}"
        )
    if not 0.0 <= args.pruning_target_reduction <= 1.0:
        raise ValueError(
            f"--pruning_target_reduction must be in [0,1], got {args.pruning_target_reduction}"
        )
    if not 0.0 <= args.importance_ema_decay <= 1.0:
        raise ValueError(
            f"--importance_ema_decay must be in [0,1], got {args.importance_ema_decay}"
        )
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError(
            f"--lora_dropout must be in [0,1), got {args.lora_dropout}"
        )
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError(
            f"--warmup_ratio must be in [0,1], got {args.warmup_ratio}"
        )
    if args.num_train_epochs < 1:
        raise ValueError(
            f"--num_train_epochs must be >= 1, got {args.num_train_epochs}"
        )
    if args.weight_decay < 0:
        raise ValueError(
            f"--weight_decay must be >= 0, got {args.weight_decay}"
        )
    if args.apply_pruning and args.pruning_steps < 1:
        raise ValueError(
            f"--pruning_steps must be >= 1 when --apply_pruning, got {args.pruning_steps}"
        )
    if args.max_grad_norm < 0:
        raise ValueError(
            f"--max_grad_norm must be >= 0 (0 disables clipping), got {args.max_grad_norm}"
        )
    if args.save_strategy == "steps" and args.save_steps < 1:
        raise ValueError(
            f"--save_steps must be >= 1 when --save_strategy=steps, got {args.save_steps}"
        )

    if args.mt_bench_output_dir is None:
        args.mt_bench_output_dir = os.path.join(args.output_dir, "mt_bench")

    return args
