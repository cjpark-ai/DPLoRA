"""LoRA configuration helpers shared by both training entry points.
``validate_lora_configuration`` fails fast on rank mismatch (silent → diverges from paper)."""
import logging
import os

import loralib as lora

from utils.common import get_layer_size, save_metrics
from .lora_optimizer import LoRAOptimizer

logger = logging.getLogger(__name__)


def optimize_lora_config(model, dataloader, args, accelerator, output_dir):
    """Run Stage-1 ILP and return ``{layer_name: optimal_r}`` (minimises estimated
    performance loss subject to ``args.lora_budget``)."""
    logger.info(f"Starting LoRA optimization with seed {args.seed}...")

    lora_optimizer = LoRAOptimizer(
        model=model,
        r_values=args.lora_r_values,
        budget=args.lora_budget,
        device=accelerator.device,
        output_dir=os.path.join(output_dir, "lora_optimization"),
        seed=args.seed,
    )

    # model.to() must run on all ranks (each owns its device), not gated by
    # main_process_first (a sync context for main-rank-only work, e.g. downloads).
    model.to(accelerator.device)
    with accelerator.main_process_first():
        r_config = lora_optimizer.optimize(dataloader)

    if accelerator.is_main_process:
        # atomic write via save_metrics (.tmp + os.replace)
        save_metrics(r_config, os.path.join(output_dir, "lora_r_config.json"))

    # Model-agnostic logging: list all layers and the overall average rank.
    logger.info("OPTIMAL LoRA RANK CONFIGURATION:")
    for layer_name, r_value in sorted(r_config.items()):
        logger.info(f"  {layer_name}: r = {r_value}")
    if r_config:
        avg_r = sum(r_config.values()) / len(r_config)
        logger.info(f"Average r across all layers: {avg_r:.2f}")

    total_params = 0
    for layer_name, r_value in r_config.items():
        if r_value > 0:
            in_f, out_f = get_layer_size(model, layer_name)
            total_params += r_value * (in_f + out_f)
    logger.info(f"Total LoRA parameters: {total_params:,}")

    return r_config


def validate_lora_configuration(r_config, replaced_layers, logger):
    """Cross-check ILP-decided ranks vs ranks actually applied; raise on any mismatch
    (bidirectional: missing or extra). Missing ``r`` key → KeyError; negative r rejected."""
    logger.info("=" * 60)
    logger.info("VALIDATING LoRA CONFIGURATION")
    logger.info("=" * 60)

    applied_layers = 0
    skipped_layers = 0
    mismatch_layers = 0
    total_params = 0

    for layer_name, layer_info in replaced_layers.items():
        expected_r = r_config.get(layer_name, 0)

        # 'r' is a required schema field
        if 'r' not in layer_info:
            raise KeyError(
                f"replaced_layers[{layer_name!r}] is missing required 'r' key"
            )
        actual_r = layer_info['r']
        applied = layer_info.get('applied', False)

        # negative r is meaningless
        if expected_r < 0 or actual_r < 0:
            raise ValueError(
                f"Negative r not allowed: {layer_name} "
                f"(expected={expected_r}, actual={actual_r})"
            )

        param_count = 0
        if 'in_features' in layer_info and 'out_features' in layer_info:
            param_count = actual_r * (layer_info['in_features'] + layer_info['out_features'])
            total_params += param_count

        status = "MATCH" if expected_r == actual_r else "MISMATCH"
        applied_status = "APPLIED" if applied else "SKIPPED"

        logger.info(f"Layer: {layer_name}")
        logger.info(f"  Expected r: {expected_r}, Actual r: {actual_r}, Status: {status}, {applied_status}")
        if param_count > 0:
            logger.info(f"  Parameters: {param_count:,}")

        if logger.level <= logging.DEBUG:
            logger.debug(f"Layer {layer_name}: Expected r={expected_r}, "
                        f"Actual r={actual_r}, Status: {status}, {applied_status}")

        if expected_r != actual_r:
            mismatch_layers += 1
            logger.warning(f"MISMATCH in layer {layer_name}: Expected r={expected_r}, Actual r={actual_r}")

        if applied:
            applied_layers += 1
        else:
            skipped_layers += 1

    # bidirectional set comparison — catch silent skip (HF PEFT-style fail-fast).
    r_keys = set(r_config.keys())
    replaced_keys = set(replaced_layers.keys())
    missing = r_keys - replaced_keys
    extra = replaced_keys - r_keys
    for name in sorted(missing):
        mismatch_layers += 1
        logger.warning(
            f"MISSING in replaced_layers: {name} "
            f"(r_config asked r={r_config[name]} but layer was not applied — "
            f"silent skip during model_setup)"
        )
    for name in sorted(extra):
        mismatch_layers += 1
        logger.warning(
            f"EXTRA in replaced_layers: {name} (not in r_config)"
        )

    logger.info(f"LoRA SUMMARY: {applied_layers} layers applied, {skipped_layers} layers skipped")
    logger.info(f"Total LoRA parameters: {total_params:,}")

    if mismatch_layers > 0:
        logger.error(f"CRITICAL: Found {mismatch_layers} layers with r-value mismatches!")
        # Fail-fast: training with an r_config that does not match the ILP
        # decision would produce results inconsistent with the paper.
        raise RuntimeError(
            f"LoRA r-value mismatches detected: {mismatch_layers} layers. "
            f"ILP-decided r_config not faithfully applied. Cannot proceed with training."
        )
    else:
        logger.info("SUCCESS: All r-values correctly applied")
    logger.info("=" * 60)

    return {
        "applied_layers": applied_layers,
        "skipped_layers": skipped_layers,
        "mismatch_layers": mismatch_layers,
        "total_params": total_params,
    }


def check_lora_merge_status(model):
    """Check LoRA merge status; anomaly ``r=0 + merged=True`` (pruned-out layer
    merged) should never occur → logged at WARNING."""
    logger.info("=" * 80)
    logger.info("CHECKING LoRA MERGE STATUS")
    logger.info("=" * 80)

    merged_count = 0
    unmerged_count = 0
    zero_r_count = 0
    anomaly_count = 0

    for name, module in model.named_modules():
        if isinstance(module, (lora.Linear, lora.MergedLinear,
                                lora.ConvLoRA, lora.Embedding)):
            if module.r == 0 and getattr(module, 'merged', False):
                anomaly_count += 1
                zero_r_count += 1
                logger.warning(
                    f"Layer {name}: r=0 but merged=True (anomaly — "
                    f"pruned-out layer should never be marked merged)"
                )
            elif module.r == 0:
                zero_r_count += 1
                logger.debug(f"Layer {name} has r=0 (pruned out)")
            elif getattr(module, 'merged', False):
                merged_count += 1
                logger.debug(f"Layer {name} is MERGED")
            else:
                unmerged_count += 1
                logger.debug(f"Layer {name} is UNMERGED")

    logger.info(
        f"Summary: Merged={merged_count}, Unmerged={unmerged_count}, "
        f"Zero-r={zero_r_count}"
        + (f", Anomalies={anomaly_count}" if anomaly_count > 0 else "")
    )
    logger.info("=" * 80)

    return merged_count, unmerged_count
