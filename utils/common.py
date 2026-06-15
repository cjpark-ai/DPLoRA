"""Generic training-time helpers shared by all entry points."""
import json
import logging
import os
import sys

import torch
import transformers

logger = logging.getLogger(__name__)


def format_runtime(seconds: float) -> str:
    return f"{seconds:.2f}"


def get_memory_stats() -> dict:
    """Peak/reserved CUDA memory (GB) per device — ``gpu{i}_*`` + legacy ``*_gb``
    mirroring device 0; {} on CPU. ``peak`` = max since last reset_peak_memory_stats()."""
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        stats[f"gpu{i}_peak_allocated_gb"] = (
            torch.cuda.max_memory_allocated(i) / (1024**3)
        )
        stats[f"gpu{i}_reserved_gb"] = (
            torch.cuda.memory_reserved(i) / (1024**3)
        )
    stats["peak_allocated_gb"] = stats["gpu0_peak_allocated_gb"]
    stats["reserved_gb"] = stats["gpu0_reserved_gb"]
    return stats


def setup_minimal_logging():
    """Keep transformers at warning level."""
    transformers.logging.set_verbosity_warning()


def validate_model_parameters(model) -> bool:
    """True iff trainable params are finite and count > 0 (zero = failure, surfaces
    silent LoRA-setup bugs: adapters not injected or over-frozen)."""
    has_problem = False
    has_trainable = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_trainable = True
            # torch.isfinite covers both NaN and Inf in one pass.
            if not torch.isfinite(param).all():
                logger.warning(f"Parameter {name} contains NaN or Inf values!")
                has_problem = True

    if not has_trainable:
        logger.error(
            "No trainable parameters found. LoRA setup likely misconfigured."
        )
        return False

    if has_problem:
        logger.warning(
            "Model contains problematic parameters. Consider adjusting learning rate."
        )

    return not has_problem


def get_layer_size(model, layer_name: str) -> tuple:
    """Return ``(in_features, out_features)`` for the named submodule,
    or ``(0, 0)`` if unresolved or not Linear-like."""
    try:
        names = layer_name.split('.')
        module = model
        for name in names:
            module = getattr(module, name)

        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            return module.in_features, module.out_features
        logger.warning(f"Layer {layer_name} has no in_features/out_features attributes")
        return 0, 0
    except (AttributeError, ValueError) as e:
        logger.warning(f"Could not resolve layer size for {layer_name}: {e}")
        return 0, 0


def save_metrics(metrics: dict, output_path: str) -> None:
    """Serialize a metrics dict to JSON atomically (tmp + fsync + os.replace).
    ``allow_nan=False`` → NaN/Inf raises; serialize fail warns+cleans, I/O fail raises."""
    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.dim() == 0 else obj.tolist()
        np_mod = sys.modules.get("numpy")
        if np_mod is not None:
            if isinstance(obj, np_mod.ndarray):
                return obj.tolist()
            if isinstance(obj, np_mod.generic):
                return obj.item()
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(
                metrics, f, indent=2, default=_to_serializable, allow_nan=False
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize metrics: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
