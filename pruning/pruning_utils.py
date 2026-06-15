"""Utility functions for LoRA pruning: layer importance, performance-loss
estimation, layer classification, and pruning validation."""

import gc
import logging
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import pruning_config

logger = logging.getLogger(__name__)


class ILPNotSolvedError(RuntimeError):
    """Raised when the ILP solver returns a non-Optimal/non-Not-Solved status — a
    fail-fast exception instead of a silent fallback to original/smallest-r config."""


def estimate_parameter_cost(
    r_value: int,
    layer_size: Tuple[int, int],
) -> float:
    """Estimate the parameter cost of a LoRA layer (Paper Eq.4: C_{l,r} = r * (d_in + d_out))."""
    in_features, out_features = layer_size
    return r_value * (in_features + out_features)


def _apply_nan_inf_replacement(importances: Dict[str, float]) -> None:
    """In-place NaN/Inf importance replacement with scale-aware sentinels: NaN →
    min(positive)*1e-9, ±Inf → max(positive)*1e9, 0 stays 0. Raises if all NaN/Inf."""
    finite_pos = [v for v in importances.values() if math.isfinite(v) and v > 0]
    has_non_finite = any(not math.isfinite(v) for v in importances.values())

    if not has_non_finite:
        return

    if not finite_pos:
        raise RuntimeError(
            "All importance values are NaN/Inf — cannot compute reference "
            "scale for replacement. Aborting to avoid silent corruption."
        )

    low_sentinel = min(finite_pos) * 1e-9
    high_sentinel = max(finite_pos) * 1e9

    # A NaN/Inf importance is a real divergence signal (overflow / exploding grad);
    # collect affected layers and emit one WARNING. Substitution unchanged (observability).
    nan_names, inf_names = [], []
    for name in importances:
        val = importances[name]
        if math.isnan(val):
            importances[name] = low_sentinel
            nan_names.append(name)
        elif math.isinf(val):
            importances[name] = high_sentinel
            inf_names.append(name)

    if nan_names or inf_names:
        logger.warning(
            "importance sanitized: %d NaN -> %.3e %s | %d Inf -> %.3e %s",
            len(nan_names), low_sentinel, nan_names,
            len(inf_names), high_sentinel, inf_names,
        )


def get_layer_size(model: nn.Module, layer_name: str) -> Tuple[int, int]:
    """Return ``(in_features, out_features)`` for a named layer, or (0, 0) if unresolved."""
    try:
        names = layer_name.split('.')
        module = model
        for name in names:
            module = getattr(module, name)

        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            return module.in_features, module.out_features

        # For merged/special layers, try to infer dimensions
        if hasattr(module, 'weight'):
            shape = module.weight.shape
            if len(shape) == 2:
                return shape[1], shape[0]  # (in_features, out_features)

        logger.warning(f"Could not determine dimensions for layer {layer_name}")
        return 0, 0
    except (AttributeError, ValueError) as e:
        logger.warning(f"Error getting size for {layer_name}: {e}")
        return 0, 0


def classify_layers_by_type(layer_names: List[str]) -> Dict[str, List[str]]:
    """Classify target layers by type — 4-way (LLaMA / BART / T5 / default BERT-RoBERTa)
    with an explicit ``"other"`` fallback (filtered out if empty)."""
    # Model detection (priority LLaMA→BART→T5→default RoBERTa): ``gate_proj`` is unique
    # to LLaMA MLPs; ``q_proj`` alone is ambiguous (BART too); ``fc1``/``fc2`` mark BART.
    is_llama = any('gate_proj' in name for name in layer_names)
    is_bart = (not is_llama) and any(
        name.endswith('fc1') or name.endswith('fc2') for name in layer_names
    )
    is_t5 = (not is_llama and not is_bart) and any(
        name.endswith('.q') or name.endswith('.wi') or name.endswith('.wi_0')
        or name.endswith('.wo')
        for name in layer_names
    )

    if is_llama:
        layer_types = {
            "q_proj": [], "k_proj": [], "v_proj": [], "o_proj": [],
            "gate_proj": [], "up_proj": [], "down_proj": [],
            "other": [],
        }
        for layer_name in layer_names:
            if layer_name.endswith('q_proj'):
                layer_types["q_proj"].append(layer_name)
            elif layer_name.endswith('k_proj'):
                layer_types["k_proj"].append(layer_name)
            elif layer_name.endswith('v_proj'):
                layer_types["v_proj"].append(layer_name)
            elif layer_name.endswith('o_proj'):
                layer_types["o_proj"].append(layer_name)
            elif layer_name.endswith('gate_proj'):
                layer_types["gate_proj"].append(layer_name)
            elif layer_name.endswith('up_proj'):
                layer_types["up_proj"].append(layer_name)
            elif layer_name.endswith('down_proj'):
                layer_types["down_proj"].append(layer_name)
            else:
                layer_types["other"].append(layer_name)

    elif is_bart:
        layer_types = {
            "q_proj": [], "k_proj": [], "v_proj": [], "out_proj": [],
            "fc1": [], "fc2": [],
            "other": [],
        }
        for layer_name in layer_names:
            if layer_name.endswith('q_proj'):
                layer_types["q_proj"].append(layer_name)
            elif layer_name.endswith('k_proj'):
                layer_types["k_proj"].append(layer_name)
            elif layer_name.endswith('v_proj'):
                layer_types["v_proj"].append(layer_name)
            elif layer_name.endswith('out_proj'):
                layer_types["out_proj"].append(layer_name)
            elif layer_name.endswith('fc1'):
                layer_types["fc1"].append(layer_name)
            elif layer_name.endswith('fc2'):
                layer_types["fc2"].append(layer_name)
            else:
                layer_types["other"].append(layer_name)

    elif is_t5:
        layer_types = {
            "q": [], "k": [], "v": [], "o": [],
            "wi": [], "wo": [],
            "other": [],
        }
        for layer_name in layer_names:
            if layer_name.endswith('.q'):
                layer_types["q"].append(layer_name)
            elif layer_name.endswith('.k'):
                layer_types["k"].append(layer_name)
            elif layer_name.endswith('.v'):
                layer_types["v"].append(layer_name)
            elif layer_name.endswith('.o'):
                layer_types["o"].append(layer_name)
            elif (layer_name.endswith('.wi') or layer_name.endswith('.wi_0')
                  or layer_name.endswith('.wi_1')):
                layer_types["wi"].append(layer_name)
            elif layer_name.endswith('.wo'):
                layer_types["wo"].append(layer_name)
            else:
                layer_types["other"].append(layer_name)

    else:
        # Default: BERT / RoBERTa
        layer_types = {
            "query": [], "key": [], "value": [],
            "attention_output": [], "intermediate": [], "output": [],
            "other": [],
        }
        for layer_name in layer_names:
            if "query" in layer_name:
                layer_types["query"].append(layer_name)
            elif "key" in layer_name:
                layer_types["key"].append(layer_name)
            elif "value" in layer_name:
                layer_types["value"].append(layer_name)
            elif "attention.output" in layer_name:
                layer_types["attention_output"].append(layer_name)
            elif "intermediate" in layer_name:
                layer_types["intermediate"].append(layer_name)
            elif "output" in layer_name and "attention" not in layer_name:
                layer_types["output"].append(layer_name)
            else:
                layer_types["other"].append(layer_name)

    filtered = {k: v for k, v in layer_types.items() if v}

    # Fallback for unrecognized architectures: if all layers landed in "other",
    # split by semantic suffix via the non-numeric path-component walkback rule.
    if list(filtered.keys()) == ["other"]:
        return _auto_classify_plus(layer_names)
    return filtered


_GENERIC_LAST_NAMES = {"dense", "proj", "linear", "fc"}
_STILL_GENERIC_TWO_COMP = {"output.dense"}


def _auto_classify_plus(layer_names: List[str]) -> Dict[str, List[str]]:
    """Architecture-agnostic fallback classifier: strips numeric path components, keys
    by the last non-numeric one, extending while it stays generic (``dense``/``proj``)."""
    groups: Dict[str, List[str]] = {}
    for n in layer_names:
        parts = [p for p in n.split(".") if not p.isdigit()]
        if not parts:
            groups.setdefault(n, []).append(n)
            continue
        key = parts[-1]
        depth = 1
        while depth < len(parts) and (
            key in _GENERIC_LAST_NAMES or key in _STILL_GENERIC_TWO_COMP
        ):
            depth += 1
            key = ".".join(parts[-depth:])
        groups.setdefault(key, []).append(n)
    return groups


def measure_layer_importance(
    model: nn.Module,
    dataloader: DataLoader,
    r_config: Dict[str, int],
    device: torch.device,
    num_batches: Optional[int] = None,
    prev_importances: Optional[Dict[str, float]] = None,
    ema_decay: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Stage 2 layer importance (Paper Eq.6/7): per sample
    sqrt(mean((grad A)^2)*mean((grad B)^2))*sqrt(r_l), averaged over mini-batches then
    EMA-blended. Returns (importances, raw_ema); pass raw_ema back as prev_importances."""
    if num_batches is None:
        num_batches = pruning_config.IMPORTANCE_NUM_BATCHES
    actual_ema_decay = ema_decay if ema_decay is not None else 0.9

    # Validate ema_decay (NaN comparisons silently pass <=0 / >=1).
    if math.isnan(actual_ema_decay) or not (0.0 <= actual_ema_decay <= 1.0):
        raise ValueError(
            f"ema_decay must be a finite value in [0, 1], got {actual_ema_decay}"
        )

    use_memory_efficient_mode = os.getenv('USE_MEMORY_EFFICIENT_IMPORTANCE', 'false').lower() == 'true'
    chunk_size = int(os.getenv('IMPORTANCE_CHUNK_SIZE', '8'))

    if use_memory_efficient_mode:
        logger.info(f"Measuring layer importance with EMA decay {actual_ema_decay} "
                    f"(memory-efficient, chunk_size={chunk_size})")
    else:
        logger.info(f"Measuring layer importance with EMA decay: {actual_ema_decay}")
    start_time = time.time()

    was_training = model.training
    model.to(device)
    model.train()

    # Optional gradient checkpointing for memory-constrained settings.
    gradient_checkpointing_was_enabled = False
    if use_memory_efficient_mode and hasattr(model, 'gradient_checkpointing_enable'):
        if hasattr(model.config, 'gradient_checkpointing') and not model.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            gradient_checkpointing_was_enabled = True

    # Back up existing grads so the importance backward doesn't overwrite training
    # gradients. Must run after model.to(device) (backup tensors share device).
    saved_grads = {n: (p.grad.clone() if p.grad is not None else None)
                   for n, p in model.named_parameters() if p.requires_grad}

    try:
        # Accumulate as on-device tensors; defer .item() to end of fn
        # (avoids a 2 * sync per sample-layer).
        importances_acc = {
            name: torch.zeros((), device=device, dtype=torch.float32)
            for name in r_config
        }

        lora_layers = {}
        for name, module in model.named_modules():
            if (hasattr(module, 'lora_A') and hasattr(module, 'lora_B')) and name in r_config:
                lora_layers[name] = module

        # Process batches sample by sample to ensure abs → mean order
        sample_count = 0

        sample_counts = {name: 0 for name in r_config}

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            if use_memory_efficient_mode and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch_size = None
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.size(0)
                    break

            if batch_size is None:
                logger.warning(f"Batch {batch_idx}: Could not determine batch size, skipping")
                continue

            for sample_idx in range(batch_size):
                sample_batch = {k: v[sample_idx:sample_idx+1] if isinstance(v, torch.Tensor) else v
                              for k, v in batch.items()}

                model.zero_grad(set_to_none=True)
                outputs = model(**sample_batch)

                loss = None
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                elif hasattr(outputs, 'logits') and outputs.logits is not None:
                    if 'labels' in sample_batch:
                        # Use cross entropy loss for classification
                        if outputs.logits.dim() > 1 and outputs.logits.size(-1) > 1:
                            loss = torch.nn.functional.cross_entropy(outputs.logits, sample_batch['labels'])
                        else:
                            # Use MSE for regression
                            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), sample_batch['labels'].float())
                    else:
                        # Fail-fast: a pseudo-loss surrogate (mean(logits**2)) is
                        # disconnected from the task loss L(x), breaking Paper Eq.7.
                        raise ValueError(
                            f"measure_layer_importance: outputs.loss is None and "
                            f"sample_batch has no 'labels' key "
                            f"(batch keys: {list(sample_batch.keys())}). "
                            f"Using a pseudo-loss (mean(logits**2)) surrogate would "
                            f"produce an importance signal disconnected from the "
                            f"paper's task loss L(x). Ensure the importance_dataloader "
                            f"yields batches containing 'labels', or that the model "
                            f"returns outputs.loss directly."
                        )

                if loss is None:
                    logger.warning(f"Batch {batch_idx}, Sample {sample_idx}: Unable to compute loss, skipping")
                    continue

                loss.backward()
                sample_count += 1

                # Paper Eq.7: I_l^update = sqrt(mean((grad A)^2) * mean((grad B)^2)) * sqrt(r_l)
                # tensor accumulation, no per-sample .item()
                for name, layer in lora_layers.items():
                    if layer.lora_A.grad is not None and layer.lora_B.grad is not None:
                        grad_a_sq = (layer.lora_A.grad ** 2).mean()
                        grad_b_sq = (layer.lora_B.grad ** 2).mean()
                        imp = (grad_a_sq * grad_b_sq).sqrt() * (r_config[name] ** 0.5)
                        importances_acc[name] = importances_acc[name] + imp
                        sample_counts[name] += 1

                if use_memory_efficient_mode and torch.cuda.is_available() and sample_idx % 4 == 0:
                    torch.cuda.empty_cache()

        # materialize tensor accumulator → Python float with one .item() per layer.
        importances = {}
        for name in r_config:
            if sample_counts[name] > 0:
                importances[name] = (importances_acc[name] / sample_counts[name]).item()
            else:
                importances[name] = 0.0

        # sanitize first (NaN→small / Inf→large sentinel; raises if all NaN/Inf),
        # then EMA blend.
        _apply_nan_inf_replacement(importances)

        # Paper Eq.6: EMA blend with the prior step. ``prev_importances`` must be the
        # raw-EMA snapshot from the prior call so both terms share scale.
        if prev_importances is not None:
            for name in importances:
                if name in prev_importances:
                    old_importance = prev_importances[name]
                    new_importance = importances[name]
                    importances[name] = actual_ema_decay * old_importance + (1 - actual_ema_decay) * new_importance

        # Snapshot raw-EMA BEFORE normalize/floor; caller stores it as next
        # prev_importances so EMA always operates on raw scale.
        raw_ema = {k: v for k, v in importances.items()}

        # Optional normalization (off by default; matches Stage 1).
        if pruning_config.NORMALIZE_IMPORTANCE and importances:
            max_importance = max(importances.values())
            if max_importance > 0:
                for name in importances:
                    importances[name] /= max_importance

        # The ILP handles zero importance.

        if pruning_config.LOG_IMPORTANCE_SCORES:
            logger.info(f"Layer importance scores (top 10):")
            for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {name}: {importance:.6f}")

        logger.info(f"Importance measurement completed in {time.time() - start_time:.2f}s")
        return importances, raw_ema

    finally:
        # Restore the training-accumulated gradients we backed up above.
        for n, p in model.named_parameters():
            if n in saved_grads:
                p.grad = saved_grads[n]

        if gradient_checkpointing_was_enabled and hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()

        if not was_training:
            model.eval()


def estimate_performance_loss(
    current_r: int,
    new_r: int,
    importance: float,
) -> float:
    """Estimate performance loss when reducing a layer's r (Paper Eq.8):
    L = I_l^(t)*(r_current-r_new)/r_current (``importance`` = Eq.6 EMA I_l^(t))."""
    if new_r >= current_r:
        return 0.0
    # Layer already pruned out, no further loss.
    if current_r <= 0:
        return 0.0

    reduction_ratio = (current_r - new_r) / current_r
    return importance * reduction_ratio


def calculate_total_parameters(r_config: Dict[str, int], layer_sizes: Dict[str, Tuple[int, int]]) -> int:
    """Total LoRA parameter count for an r_config (sum of r * (in + out))."""
    total_params = 0

    for layer_name, r in r_config.items():
        if r <= 0:
            continue

        if layer_name in layer_sizes:
            in_features, out_features = layer_sizes[layer_name]
            layer_params = r * (in_features + out_features)
            total_params += layer_params

    return total_params


def per_type_feasibility_floor(
    layer_names: List[str],
    layer_sizes: Dict[str, Tuple[int, int]],
    r_config: Dict[str, int],
    available_r_values: List[int],
) -> int:
    """Minimum total LoRA param cost satisfying per-type sum(r)>=1 (Paper Eq.5):
    cheapest selectable rank on the cheapest layer per type, summed. Budget below
    this floor makes the ILP infeasible (raises ILPNotSolvedError)."""
    floor = 0
    for _type, layers in classify_layers_by_type(layer_names).items():
        best = None
        for l in layers:
            if l not in layer_sizes:
                continue
            r_opts = [r for r in available_r_values if 1 <= r <= r_config.get(l, 0)]
            if not r_opts:
                continue
            in_f, out_f = layer_sizes[l]
            c = min(r_opts) * (in_f + out_f)
            best = c if best is None else min(best, c)
        if best is not None:
            floor += best
    return floor


def get_lora_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Return ``{name: module}`` for all LoRA modules in the model."""
    lora_modules = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_modules[name] = module
    return lora_modules


def save_model_checkpoint(model: nn.Module, path: str) -> None:
    """Save LoRA-only weights for rollback (minimizes checkpoint size)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lora_state_dict = {}
    for name, param in model.state_dict().items():
        if 'lora_' in name:
            lora_state_dict[name] = param

    torch.save(lora_state_dict, path)
    logger.info(f"Saved LoRA checkpoint to {path}")


def load_model_checkpoint(model: nn.Module, path: str, allow_size_mismatch: bool = True) -> nn.Module:
    """Load LoRA weights for rollback; ``allow_size_mismatch`` adapts r-changed layers."""
    if not os.path.exists(path):
        logger.error(f"Checkpoint not found at {path}")
        return model

    try:
        lora_state_dict = torch.load(path, map_location='cpu')

        model_state_dict = model.state_dict()

        resized_params = []

        mismatched_params = []
        for name, checkpoint_param in lora_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].shape != checkpoint_param.shape:
                    mismatched_params.append((name, checkpoint_param.shape, model_state_dict[name].shape))

        if mismatched_params and not allow_size_mismatch:
            logger.error(f"Size mismatches detected in {len(mismatched_params)} parameters.")
            for name, checkpoint_shape, model_shape in mismatched_params:
                logger.error(f"  {name}: checkpoint={checkpoint_shape}, model={model_shape}")

            logger.error("Rollback aborted due to parameter size mismatches. Use allow_size_mismatch=True to attempt resize.")
            return model

        for name, checkpoint_param in lora_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == checkpoint_param.shape:
                    model_state_dict[name] = checkpoint_param
                elif allow_size_mismatch and ('lora_A' in name or 'lora_B' in name):
                    # dim sanity check — LoRA params must be 2-d; a corrupt 3-d/scalar
                    # tensor would silently overwrite .data and break forward passes.
                    if checkpoint_param.dim() != 2:
                        logger.error(
                            f"Skip {name}: corrupt checkpoint param has dim "
                            f"{checkpoint_param.dim()} (expected 2 for LoRA A/B). "
                            f"shape={tuple(checkpoint_param.shape)}"
                        )
                        continue
                    # True rollback: model conforms to checkpoint shape; in-place .data
                    # preserves Parameter id (caller must resync m,v via resync_optimizer_with_model).
                    logger.warning(f"Resizing parameter {name}: checkpoint={checkpoint_param.shape}, model={model_state_dict[name].shape}")

                    module_path = '.'.join(name.split('.')[:-1])
                    param_name = name.split('.')[-1]

                    try:
                        module = model
                        for part in module_path.split('.'):
                            module = getattr(module, part)

                        # checkpoint_r is the rank we are rolling back to
                        if param_name == 'lora_A':
                            checkpoint_r = checkpoint_param.shape[0]
                        else:  # 'lora_B'
                            checkpoint_r = checkpoint_param.shape[1]

                        current_param = getattr(module, param_name)
                        new_data = checkpoint_param.detach().to(
                            dtype=current_param.dtype, device=current_param.device
                        ).clone()

                        # In-place .data replacement (preserves Parameter object id)
                        with torch.no_grad():
                            current_param.data = new_data

                        # Sync model_state_dict so subsequent model.load_state_dict has matching shape
                        model_state_dict[name] = new_data
                        resized_params.append(name)

                        # True rollback: r/scaling reflect checkpoint, not current
                        if hasattr(module, 'r') and hasattr(module, 'scaling') and hasattr(module, 'lora_alpha'):
                            module.r = checkpoint_r
                            module.scaling = module.lora_alpha / checkpoint_r if checkpoint_r > 0 else 0

                    except (AttributeError, ValueError) as e:
                        logger.error(f"Error resizing {name}: {e}")
                else:
                    logger.warning(f"Skipping parameter {name} due to shape mismatch: checkpoint={checkpoint_param.shape}, model={model_state_dict[name].shape}")

        model.load_state_dict(model_state_dict)

        if resized_params:
            logger.info(f"Resized {len(resized_params)} LoRA parameters during checkpoint loading")

        logger.info(f"Loaded LoRA checkpoint from {path}")

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return model


def modify_lora_layers(
    model: nn.Module,
    new_r_config: Dict[str, int]
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Modify LoRA layers to ``new_r_config``; return (model, changes)."""
    original_level = logger.level
    original_propagate = getattr(logger, 'propagate', True)

    logger.propagate = True
    logger.setLevel(logging.INFO)

    temp_handler = None
    original_handler_levels = {}

    if not logger.handlers:
        temp_handler = logging.StreamHandler(sys.stdout)
        temp_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        temp_handler.setFormatter(formatter)
        logger.addHandler(temp_handler)
    else:
        for handler in logger.handlers:
            original_handler_levels[handler] = handler.level
            handler.setLevel(logging.INFO)

    try:
        changes = {}
        lora_modules = get_lora_modules(model)
        mismatch_detected = False

        # Merge-state guard: if a LoRA module is merged (W = W_orig + BA*s), in-place
        # lora_A/B replace + train() corrupts W to a residual leak. Fail-fast (always train).
        for _name, _m in lora_modules.items():
            if getattr(_m, 'merged', False):
                raise AssertionError(
                    f"modify_lora_layers called while {_name} is in merged state. "
                    f"This would corrupt base weight on next train→eval transition. "
                    f"Call model.train() to unmerge before modifying LoRA layers."
                )

        logger.info("=" * 60)
        logger.info("MODIFYING LoRA LAYERS")
        logger.info("=" * 60)

        layer_names = list(new_r_config.keys())
        layer_types = classify_layers_by_type(layer_names)

        total_decreased = 0
        total_increased = 0
        total_unchanged = 0
        total_params_before = 0
        total_params_after = 0

        for layer_type, layers in layer_types.items():
            logger.info(f"\n[{layer_type.upper()} LAYERS]")

            for layer_name in sorted(layers):
                new_r = new_r_config[layer_name]

                if layer_name not in lora_modules:
                    if new_r == 0:
                        logger.info(f"Layer {layer_name}: r=0 (SKIPPED)")
                        total_unchanged += 1
                    else:
                        logger.warning(f"Layer {layer_name} not found as a LoRA module (r={new_r}), SKIPPING...")
                    continue

                module = lora_modules[layer_name]
                current_r = module.r if hasattr(module, 'r') else getattr(module, 'lora_A').shape[0]
                new_r = new_r_config[layer_name]

                in_features = module.lora_A.size(1)
                out_features = module.lora_B.size(0)
                current_params = current_r * (in_features + out_features)
                new_params = new_r * (in_features + out_features)

                total_params_before += current_params

                if current_r == new_r:
                    logger.info(f"Layer {layer_name}: r={current_r} (UNCHANGED)")
                    total_unchanged += 1
                    total_params_after += current_params
                    continue

                changes[layer_name] = {
                    'old_r': current_r,
                    'new_r': new_r,
                    'in_features': in_features,
                    'out_features': out_features,
                    'change_type': None,    # filled per branch below; asserted before SUMMARY
                    'top_indices': None,    # populated only on rank decrease
                }

                if new_r < current_r:
                    total_decreased += 1
                else:
                    total_increased += 1

                if new_r == 0:
                    # Branch r>0 → r=0 (in-place fill_(0), shape preserved). Reached by
                    # monotonic pruning (Eq.11); also by rollback or relaxed constraint.
                    with torch.no_grad():
                        module.lora_A.fill_(0)
                        module.lora_B.fill_(0)
                        module.r = new_r
                        module.scaling = module.lora_alpha / new_r if new_r > 0 else 0

                    changes[layer_name]['change_type'] = 'zero'
                    logger.info(f"Layer {layer_name}: r={current_r} → 0 (zeroed)")
                    logger.info(f"  Parameters: {current_params:,} → 0 ({-current_params:,})")
                    continue

                if current_r == 0:
                    # Branch r=0 → r>0 (kaiming fresh). NOT reachable under monotonic pruning
                    # (Eq.11). Gotcha: a shape-match may carry stale Adam moments onto fresh weights.
                    with torch.no_grad():
                        new_lora_A = torch.zeros(
                            (new_r, module.lora_A.size(1)),
                            device=module.lora_A.device,
                            dtype=module.lora_A.dtype
                        )
                        new_lora_B = torch.zeros(
                            (module.lora_B.size(0), new_r),
                            device=module.lora_B.device,
                            dtype=module.lora_B.dtype
                        )

                        nn.init.kaiming_uniform_(new_lora_A, a=math.sqrt(5))

                        module.lora_A = nn.Parameter(new_lora_A)
                        module.lora_B = nn.Parameter(new_lora_B)
                        module.r = new_r
                        module.scaling = module.lora_alpha / new_r

                    changes[layer_name]['change_type'] = 'reinit'
                    logger.info(f"Layer {layer_name}: r=0 → {new_r} (reinitialized)")
                    logger.info(f"  Parameters: 0 → {new_params:,} ({new_params:+,})")

                    total_params_after += new_params
                    continue

                with torch.no_grad():
                    if new_r < current_r:
                        # Paper Eq.15: S_{l,i} = ||A_l[:, i]||_2 * ||B_l[i, :]||_2
                        # (loralib transpose convention: lora_A is (r, in), lora_B is (out, r))
                        a_dim_norms = torch.norm(module.lora_A, dim=1)
                        b_dim_norms = torch.norm(module.lora_B, dim=0)
                        dim_importance_scores = a_dim_norms * b_dim_norms

                        # Paper Eq.16: top-k indices, k = r_new
                        _, top_indices = torch.topk(dim_importance_scores, new_r)

                        new_lora_A = module.lora_A[top_indices]
                        new_lora_B = module.lora_B[:, top_indices]

                        changes[layer_name]['change_type'] = 'decrease'
                        changes[layer_name]['top_indices'] = top_indices.detach().cpu().tolist()
                        logger.info(f"Layer {layer_name}: r={current_r} → {new_r} (PRUNED BY {current_r-new_r})")
                    else:
                        # Branch B>A: rank EXPAND. NOT reachable under monotonic pruning
                        # (Eq.11). Adam moments zero-padded via _adapt_lora_state('increase').
                        new_lora_A = torch.zeros(
                            (new_r, module.lora_A.size(1)),
                            device=module.lora_A.device,
                            dtype=module.lora_A.dtype
                        )
                        new_lora_B = torch.zeros(
                            (module.lora_B.size(0), new_r),
                            device=module.lora_B.device,
                            dtype=module.lora_B.dtype
                        )

                        new_lora_A[:current_r].copy_(module.lora_A)
                        new_lora_B[:, :current_r].copy_(module.lora_B)

                        if new_r > current_r:
                            nn.init.kaiming_uniform_(new_lora_A[current_r:], a=math.sqrt(5))

                        changes[layer_name]['change_type'] = 'increase'
                        logger.info(f"Layer {layer_name}: r={current_r} → {new_r} (expanded by {new_r-current_r})")

                    module.lora_A = nn.Parameter(new_lora_A)
                    module.lora_B = nn.Parameter(new_lora_B)
                    module.r = new_r
                    module.scaling = module.lora_alpha / new_r

                    # Verify that r was actually updated
                    if module.r != new_r:
                        logger.error(f"FAILED to update r value for {layer_name}: expected {new_r}, got {module.r}")
                        mismatch_detected = True

                    logger.info(f"  Parameters: {current_params:,} → {new_params:,} ({new_params-current_params:+,})")

                    total_params_after += new_params

        # Validate change_type was set on every recorded change (catches branch additions
        # that forget to label themselves; otherwise resync would silently drop state).
        for _layer_name, _change in changes.items():
            assert _change.get('change_type') in ('zero', 'reinit', 'decrease', 'increase'), \
                f"Missing or invalid change_type for {_layer_name}: " \
                f"{_change.get('change_type')}"

        total_reduction = total_params_before - total_params_after
        reduction_pct = (total_reduction / max(total_params_before, 1)) * 100

        logger.info("\nSUMMARY OF CHANGES:")
        logger.info(f"  Decreased r-values: {total_decreased} layers")
        logger.info(f"  Increased r-values: {total_increased} layers")
        logger.info(f"  Unchanged r-values: {total_unchanged} layers")
        logger.info(f"  Total parameters: {total_params_before:,} → {total_params_after:,} ({total_params_after-total_params_before:+,})")
        logger.info(f"  Parameter reduction: {reduction_pct:.2f}%")

        if mismatch_detected:
            logger.error("CRITICAL: Some r-values were not correctly updated in the model.")

        logger.info("=" * 60)

        return model, changes

    finally:
        if temp_handler:
            logger.removeHandler(temp_handler)
            temp_handler = None

        for handler, level in original_handler_levels.items():
            handler.setLevel(level)

        logger.setLevel(original_level)
        logger.propagate = original_propagate


def evaluate_model(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate the model, returning accuracy (classification) or -mean_loss."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                outputs = model(**batch)

                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss += outputs.loss.item()

                if hasattr(outputs, 'logits') and 'labels' in batch:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
    finally:
        # Restore training mode: loralib model.eval() merges (merged=True, training=False);
        # without restoring, the next train step hits the merged branch (loss has no grad_fn).
        if was_training:
            model.train()

    # fail-fast on empty eval_dataloader (would otherwise be ZeroDivision).
    if total > 0:
        return correct / total
    if len(eval_dataloader) > 0:
        return -total_loss / len(eval_dataloader)
    raise ValueError(
        "evaluate_model: eval_dataloader yielded 0 batches and 0 samples; "
        "cannot compute accuracy or loss-based metric."
    )


def validate_pruning(
    model: nn.Module,
    eval_dataloader: DataLoader,
    baseline_performance: float,
    threshold: float = pruning_config.PERFORMANCE_DROP_THRESHOLD,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[float, bool]:
    """Re-evaluate after pruning; return (performance, is_acceptable) where
    acceptable = drop <= threshold * baseline."""
    logger.info(f"Validating pruned model (baseline: {baseline_performance:.4f}, threshold: {threshold:.4f})")

    performance = evaluate_model(model, eval_dataloader, device)

    performance_drop = max(0, baseline_performance - performance)
    performance_drop_percentage = (performance_drop / baseline_performance) * 100 if baseline_performance > 0 else 0

    is_acceptable = performance_drop <= threshold * baseline_performance

    logger.info(f"Validation results: performance={performance:.4f}, drop={performance_drop:.4f} ({performance_drop_percentage:.2f}%)")
    logger.info(f"Verdict: {'ACCEPTABLE' if is_acceptable else 'REJECTED'}")

    return performance, is_acceptable


def validate_pruning_configuration(
    model: nn.Module,
    prev_r_config: Dict[str, int],
    new_r_config: Dict[str, int],
    layer_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Validate the pruned model's config (pruning analogue of
    validate_lora_configuration); return (layer_details, summary)."""
    if logger is None:
        logger = logging.getLogger(__name__)

    original_level = logger.level
    original_propagate = getattr(logger, 'propagate', True)

    logger.propagate = True

    temp_handler = None
    original_handler_levels = {}

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        temp_handler = logging.StreamHandler(sys.stdout)
        temp_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        temp_handler.setFormatter(formatter)
        logger.addHandler(temp_handler)
    else:
        for handler in logger.handlers:
            original_handler_levels[handler] = handler.level
            handler.setLevel(logging.INFO)

    try:
        logger.info("=" * 60)
        logger.info("VALIDATING PRUNED LoRA CONFIGURATION")
        logger.info("=" * 60)

        layer_details = {}
        applied_count = 0
        mismatch_count = 0
        pruned_count = 0
        total_params_before = 0
        total_params_after = 0
        total_reduction = 0

        if layer_sizes is None:
            layer_sizes = {}
            for layer_name in set(prev_r_config.keys()).union(set(new_r_config.keys())):
                layer_sizes[layer_name] = get_layer_size(model, layer_name)

        all_layers = set(prev_r_config.keys()).union(set(new_r_config.keys()))

        layer_types = classify_layers_by_type(list(all_layers))

        for layer_type, type_layers in layer_types.items():
            logger.info(f"\n[{layer_type.upper()} LAYERS]")

            for layer_name in sorted(type_layers):
                prev_r = prev_r_config.get(layer_name, 0)
                expected_r = new_r_config.get(layer_name, 0)

                actual_r = 0
                try:
                    module = None
                    names = layer_name.split('.')
                    module = model
                    for name in names:
                        module = getattr(module, name)

                    if hasattr(module, 'r'):
                        actual_r = module.r
                    elif hasattr(module, 'lora_A'):
                        actual_r = module.lora_A.shape[0]
                except (AttributeError, ValueError):
                    logger.warning(f"Module {layer_name} not found in model")
                    continue

                in_features, out_features = layer_sizes.get(layer_name, (0, 0))
                prev_params = prev_r * (in_features + out_features) if prev_r > 0 else 0
                current_params = actual_r * (in_features + out_features) if actual_r > 0 else 0
                param_reduction = prev_params - current_params

                was_pruned = prev_r > actual_r
                if was_pruned:
                    pruned_count += 1

                is_match = actual_r == expected_r
                if not is_match:
                    mismatch_count += 1

                status = "✓ MATCH" if is_match else "✗ MISMATCH"

                if was_pruned:
                    pruned_status = "PRUNED"
                    change_symbol = "↓"
                elif prev_r < actual_r:
                    pruned_status = "INCREASED"
                    change_symbol = "↑"
                else:
                    pruned_status = "UNCHANGED"
                    change_symbol = "="

                logger.info(f"Layer: {layer_name}")
                logger.info(f"  r-value: {prev_r} {change_symbol} {actual_r} (Expected: {expected_r}) - Status: {status}, {pruned_status}")
                if prev_params > 0 or current_params > 0:
                    param_change = current_params - prev_params
                    logger.info(f"  Parameters: {prev_params:,} → {current_params:,} ({param_change:+,})")

                layer_details[layer_name] = {
                    'prev_r': prev_r,
                    'actual_r': actual_r,
                    'expected_r': expected_r,
                    'is_match': is_match,
                    'was_pruned': was_pruned,
                    'prev_params': prev_params,
                    'current_params': current_params,
                    'param_reduction': param_reduction
                }

                total_params_before += prev_params
                total_params_after += current_params
                total_reduction += param_reduction

                if actual_r > 0:
                    applied_count += 1

        reduction_percentage = (total_reduction / max(total_params_before, 1)) * 100

        summary = {
            'total_layers': len(all_layers),
            'pruned_layers': pruned_count,
            'mismatch_layers': mismatch_count,
            'applied_layers': applied_count,
            'total_params_before': total_params_before,
            'total_params_after': total_params_after,
            'total_reduction': total_reduction,
            'reduction_percentage': reduction_percentage
        }

        logger.info("-" * 60)
        logger.info(f"PRUNING SUMMARY")
        logger.info(f"  Total layers: {len(all_layers)}")
        logger.info(f"  Pruned layers: {pruned_count}")
        logger.info(f"  Applied layers: {applied_count}")
        logger.info(f"  LoRA parameters: {total_params_before:,} → {total_params_after:,} ({total_params_after - total_params_before:+,})")
        logger.info(f"  Reduction: {reduction_percentage:.2f}%")

        if mismatch_count > 0:
            logger.warning(f"CRITICAL: Found {mismatch_count} layers with r-value mismatches!")
        else:
            logger.info("SUCCESS: All r-values correctly applied")
        logger.info("=" * 60)

        return layer_details, summary

    finally:
        if temp_handler:
            logger.removeHandler(temp_handler)

        for handler, level in original_handler_levels.items():
            handler.setLevel(level)

        logger.setLevel(original_level)
        logger.propagate = original_propagate


def plot_pruning_results(
    pruning_history: List[Dict[str, Any]],
    output_dir: str,
    file_prefix: str = "pruning"
) -> None:
    """Plot pruning results (parameter reduction + performance) to ``output_dir``."""
    if not pruning_history:
        logger.warning("No pruning history to plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    steps = list(range(len(pruning_history)))
    param_counts = [step['param_count'] for step in pruning_history]
    performances = [step['performance'] for step in pruning_history]

    initial_params = param_counts[0]
    param_reductions = [(initial_params - p) / initial_params * 100 for p in param_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, param_reductions, 'bo-', linewidth=2)
    plt.xlabel('Pruning Step')
    plt.ylabel('Parameter Reduction (%)')
    plt.title('Progressive Pruning: Parameter Reduction')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_param_reduction.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, performances, 'ro-', linewidth=2)
    plt.xlabel('Pruning Step')
    plt.ylabel('Performance Metric')
    plt.title('Progressive Pruning: Model Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_performance.png"))
    plt.close()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Pruning Step')
    ax1.set_ylabel('Parameter Reduction (%)', color='blue')
    ax1.plot(steps, param_reductions, 'bo-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Performance Metric', color='red')
    ax2.plot(steps, performances, 'ro-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Progressive Pruning: Parameter Reduction vs Performance')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_combined.png"))
    plt.close()

    has_efficiency = all('efficiency' in step for step in pruning_history)
    if has_efficiency:
        efficiencies = [step.get('efficiency', 0) for step in pruning_history]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, efficiencies, 'go-', linewidth=2)
        plt.xlabel('Pruning Step')
        plt.ylabel('Performance/Parameters Ratio')
        plt.title('Progressive Pruning: Efficiency Metric')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_efficiency.png"))
        plt.close()

    logger.info(f"Pruning result plots saved to {output_dir}")


def _snapshot_optimizer_state(optimizer, model, clone=False):
    """Snapshot optimizer state by param name BEFORE modify_lora_layers (id-based
    reverse lookup needs old params live). clone=True freezes it for rollback."""
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    snap = {}
    for p in optimizer.param_groups[0]['params']:
        name = name_by_id.get(id(p))
        if name is None:
            continue
        state = optimizer.state.get(p, {})
        if clone:
            state = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                     for k, v in state.items()}
        snap[name] = {
            'shape': tuple(p.shape),
            'state': state,
        }
    return snap


def _restore_optimizer_state(optimizer, model, snapshot):
    """Restore group-0 membership + Adam state from a name-keyed snapshot (rollback;
    shapes already match, so no adaptation)."""
    old_group0 = list(optimizer.param_groups[0]['params'])
    new_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and name in snapshot]
    optimizer.param_groups[0]['params'] = new_params
    for _p in old_group0:
        optimizer.state.pop(_p, None)
    by_name = {name: p for name, p in model.named_parameters() if p.requires_grad}
    for name, entry in snapshot.items():
        p = by_name.get(name)
        if p is not None and entry.get('state'):
            optimizer.state[p] = entry['state']


def _adapt_lora_state(old_state, attr, layer_change, new_param):
    """Adapt Adam state to a new param shape after pruning. attr 'A' rank dim=0,
    'B' rank dim=1. Returns adapted dict, or {} to drop (reinit)."""
    change_type = layer_change.get('change_type')
    if change_type in ('reinit',):
        return {}  # no useful prior state

    out = {}
    for key, val in old_state.items():
        # Pass through non-tensors AND 0-dim tensors (e.g. AdamW's `step`
        # is a scalar tensor in modern PyTorch, has no rank dim to slice).
        if not torch.is_tensor(val) or val.dim() == 0:
            out[key] = val
            continue

        if change_type == 'decrease':
            top = torch.tensor(layer_change['top_indices'],
                               device=val.device, dtype=torch.long)
            if attr == 'A':
                out[key] = val.index_select(0, top).contiguous()
            else:  # 'B'
                out[key] = val.index_select(1, top).contiguous()

        elif change_type == 'increase':
            new_t = torch.zeros_like(new_param)
            old_r = layer_change['old_r']
            if attr == 'A':
                new_t[:old_r].copy_(val)
            else:
                new_t[:, :old_r].copy_(val)
            out[key] = new_t

        else:
            return {}

    return out


def resync_optimizer_with_model(optimizer, model, changes, snapshot):
    """Re-sync optimizer params/state AFTER modify_lora_layers (with its `changes`
    + the pre-pruning `snapshot`). Adam per change: same-shape carry / decrease slice
    on rank dim / increase zero-pad / reinit drop / zero carry."""
    # Only replace param_groups[0] (snapshot keys = group-0 membership); other groups
    # (AdamW no_decay: bias/LayerNorm) left untouched to avoid double-registration.
    new_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and name in snapshot]
    # Capture the OLD group-0 Parameter objects before replacing the list, so we
    # can drop their stale optimizer state surgically instead of wiping all state.
    _old_group0 = list(optimizer.param_groups[0]['params'])
    optimizer.param_groups[0]['params'] = new_params

    n_total = n_preserved = n_sliced = n_padded = n_dropped = 0
    new_state = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if name not in snapshot:
            n_dropped += 1
            continue
        old_shape = snapshot[name]['shape']
        old_st = snapshot[name]['state']

        if tuple(p.shape) == old_shape:
            new_state[p] = old_st
            n_preserved += 1
            continue

        # Shape changed -- locate change record by stripping ".lora_A"/".lora_B"
        if name.endswith('.lora_A'):
            module_name, attr = name[:-len('.lora_A')], 'A'
        elif name.endswith('.lora_B'):
            module_name, attr = name[:-len('.lora_B')], 'B'
        else:
            n_dropped += 1
            continue

        layer_change = (changes or {}).get(module_name)
        if layer_change is None:
            n_dropped += 1
            continue

        adapted = _adapt_lora_state(old_st, attr, layer_change, p)
        if adapted:
            new_state[p] = adapted
            if layer_change['change_type'] == 'decrease':
                n_sliced += 1
            elif layer_change['change_type'] == 'increase':
                n_padded += 1
            else:
                n_preserved += 1
        else:
            n_dropped += 1

    # optimizer.state is ONE dict shared across param_groups, so .clear() would wipe
    # the no_decay group too. Drop only the rebuilt group-0 params' state, then update.
    for _p in _old_group0:
        optimizer.state.pop(_p, None)
    optimizer.state.update(new_state)
    logger.warning(
        f"Optimizer resync: {n_total} params total | "
        f"{n_preserved} preserved | {n_sliced} sliced | "
        f"{n_padded} padded | {n_dropped} dropped"
    )