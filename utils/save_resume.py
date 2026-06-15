"""Save/Resume infrastructure. A checkpoint dir holds lora_state / optimizer / scheduler /
rng_state / trainer_state / pruning_state (9 keys). ``save_strategy="no"`` (default) skips it."""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


_SAFE_GLOBALS_REGISTERED = False


def _register_safe_globals_once() -> None:
    """Register classes for ``torch.load(weights_only=True)`` of our ckpts; called lazily
    at ``load_full_state``. Add a class here if a future ckpt needs it."""
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    safe = [np.ndarray, np.dtype]
    # numpy >= 2 splits dtype into many subclasses (np.dtypes.UInt32DType, ...).
    try:
        import numpy.dtypes as _np_dtypes
        for _name in dir(_np_dtypes):
            _cls = getattr(_np_dtypes, _name, None)
            if isinstance(_cls, type) and issubclass(_cls, np.dtype):
                safe.append(_cls)
    except Exception:
        pass
    # numpy array reconstruction (needed for np.random.get_state() arrays).
    try:
        from numpy.core.multiarray import _reconstruct as _np_reconstruct
        safe.append(_np_reconstruct)
    except Exception:
        pass
    try:
        from numpy._core.multiarray import _reconstruct as _np_reconstruct2
        safe.append(_np_reconstruct2)
    except Exception:
        pass
    # collections.OrderedDict appears in some state_dicts (older torch).
    try:
        from collections import OrderedDict
        safe.append(OrderedDict)
    except Exception:
        pass
    try:
        torch.serialization.add_safe_globals(safe)
    except Exception as e:
        logger.warning(f"add_safe_globals failed (continuing): {e}")
    _SAFE_GLOBALS_REGISTERED = True


# 9 keys that fully describe ``ProgressivePruningManager`` mutable state.
PRUNING_STATE_KEYS = (
    "current_r_config",
    "current_performance",
    "baseline_performance",
    "pruning_history",
    "prev_importances",
    "last_changes",
    "scheduler_current_step_idx",
    "scheduler_in_recovery_mode",
    "scheduler_recovery_step_counter",
)


# Pruning state JSON I/O
def _manager_to_dict(manager) -> dict:
    return {
        "current_r_config": dict(manager.current_r_config),
        "current_performance": manager.current_performance,
        "baseline_performance": manager.baseline_performance,
        "pruning_history": list(manager.pruning_history),
        "prev_importances": (
            None if manager.prev_importances is None
            else {k: float(v) for k, v in manager.prev_importances.items()}
        ),
        "last_changes": dict(manager.last_changes),
        "scheduler_current_step_idx": manager.scheduler.current_step_idx,
        "scheduler_in_recovery_mode": bool(manager.scheduler.in_recovery_mode),
        "scheduler_recovery_step_counter": int(manager.scheduler.recovery_step_counter),
    }


def save_pruning_state_json(manager, path: str) -> None:
    state = _manager_to_dict(manager)
    missing = set(PRUNING_STATE_KEYS) - set(state.keys())
    if missing:
        raise RuntimeError(f"_manager_to_dict is missing keys: {missing}")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_pruning_state_json(manager, path: str) -> None:
    """Restore 9 PRUNING_STATE_KEYS into ``manager`` from JSON. If loaded current_performance
    is None but baseline is set, fall back to baseline (saved before first eval)."""
    with open(path, "r") as f:
        state = json.load(f)
    missing = set(PRUNING_STATE_KEYS) - set(state.keys())
    if missing:
        raise RuntimeError(f"pruning_state.json is missing keys: {missing}")

    manager.current_r_config = dict(state["current_r_config"])
    manager.current_performance = state["current_performance"]
    manager.baseline_performance = state["baseline_performance"]
    manager.pruning_history = list(state["pruning_history"])
    manager.prev_importances = (
        None if state["prev_importances"] is None
        else {k: float(v) for k, v in state["prev_importances"].items()}
    )
    manager.last_changes = dict(state["last_changes"])
    manager.scheduler.current_step_idx = int(state["scheduler_current_step_idx"])
    manager.scheduler.in_recovery_mode = bool(state["scheduler_in_recovery_mode"])
    manager.scheduler.recovery_step_counter = int(state["scheduler_recovery_step_counter"])

    if manager.baseline_performance is not None and manager.current_performance is None:
        manager.current_performance = manager.baseline_performance


# Full checkpoint I/O
def _rng_state_dict() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def _lora_only_state_dict(model, current_r_config: Optional[dict] = None) -> dict:
    """Params to persist for bit-exact resume: LoRA (dropping r=0 to match resume shapes)
    PLUS trained/run-specific non-LoRA. classifier/pooler head MUST be saved (random under --no-train)."""
    full = model.state_dict()

    def _layer_of(param_name: str) -> str:
        return param_name.rsplit(".lora_", 1)[0]

    if current_r_config is None:
        out = {n: t for n, t in full.items() if "lora_" in n}
    else:
        out = {
            n: t for n, t in full.items()
            if "lora_" in n and current_r_config.get(_layer_of(n), 0) > 0
        }

    # Non-LoRA params that are trained or are the run-specific classifier/pooler
    # head (mirror the train loop's best_state snapshot).
    for n, p in model.named_parameters():
        if ("lora_" not in n
                and (p.requires_grad or "classifier" in n or "pooler" in n)
                and n in full):
            out[n] = full[n]
    return out


def _filtered_optimizer_state(optimizer, model, current_r_config: Optional[dict] = None) -> dict:
    """Return ``optimizer.state_dict()`` with r=0 LoRA params filtered out — a model
    rebuilt on resume has no LoRA there, so drop those entries to match it."""
    state_dict = optimizer.state_dict()
    if current_r_config is None:
        return state_dict

    def _layer_of(param_name: str) -> str:
        return param_name.rsplit(".lora_", 1)[0]

    name_by_id = {}
    for name, p in model.named_parameters():
        name_by_id[id(p)] = name

    # Mark global param indices (across all groups, matching state_dict
    # indexing): drop r=0 LoRA params, keep the rest.
    keep_old_indices = []
    drop_old_indices = set()
    global_idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            pname = name_by_id.get(id(p))
            drop = False
            if pname is not None and ".lora_" in pname:
                layer = _layer_of(pname)
                if current_r_config.get(layer, 0) == 0:
                    drop = True
            if drop:
                drop_old_indices.add(global_idx)
            else:
                keep_old_indices.append(global_idx)
            global_idx += 1

    if not drop_old_indices:
        return state_dict

    old_to_new = {old: new for new, old in enumerate(keep_old_indices)}
    new_state = {}
    for old_i, st in state_dict["state"].items():
        if old_i in old_to_new:
            new_state[old_to_new[old_i]] = st

    new_param_groups = []
    cursor = 0
    for group in state_dict["param_groups"]:
        new_group = {k: v for k, v in group.items() if k != "params"}
        new_params = []
        for old_i in group["params"]:
            if old_i in old_to_new:
                new_params.append(old_to_new[old_i])
        new_group["params"] = new_params
        new_param_groups.append(new_group)

    logger.info(
        f"Filtered optimizer state: dropped {len(drop_old_indices)} r=0 LoRA "
        f"params, kept {len(keep_old_indices)} params."
    )
    return {"state": new_state, "param_groups": new_param_groups}


def save_full_state(
    checkpoint_dir: str,
    model,
    optimizer,
    lr_scheduler,
    completed_steps: int,
    epoch: int,
    pruning_manager,
    num_update_steps_per_epoch: Optional[int] = None,
    dataloader_generator_state=None,
    dataloader_generator_live_state=None,
) -> None:
    """Save full checkpoint (≤7 files). ``dataloader_generator_state`` = epoch-start g
    (resume re-draws shuffle); ``*_live_state`` = live g re-applied after the resumed perm."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    current_r_config = (
        pruning_manager.current_r_config if pruning_manager is not None else None
    )
    torch.save(
        _lora_only_state_dict(model, current_r_config=current_r_config),
        os.path.join(checkpoint_dir, "lora_state.pt"),
    )
    torch.save(
        _filtered_optimizer_state(optimizer, model, current_r_config=current_r_config),
        os.path.join(checkpoint_dir, "optimizer.pt"),
    )
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    torch.save(_rng_state_dict(), os.path.join(checkpoint_dir, "rng_state.pt"))
    trainer_state = {"completed_steps": int(completed_steps), "epoch": int(epoch)}
    if num_update_steps_per_epoch is not None:
        trainer_state["num_update_steps_per_epoch"] = int(num_update_steps_per_epoch)
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f, indent=2)
    if pruning_manager is not None:
        save_pruning_state_json(pruning_manager, os.path.join(checkpoint_dir, "pruning_state.json"))
    if dataloader_generator_state is not None:
        torch.save(
            {
                "generator_state": dataloader_generator_state,
                "generator_live_state": dataloader_generator_live_state,
            },
            os.path.join(checkpoint_dir, "dataloader_rng.pt"),
        )

    # Per-layer LoRA alpha (scaling=alpha/r, a float attr NOT in state_dict): a rebuilt
    # model defaults to alpha=2*r → wrong scaling; persist alpha so resume restores it.
    lora_meta = {
        name: float(m.lora_alpha)
        for name, m in model.named_modules()
        if hasattr(m, "lora_A") and getattr(m, "r", 0) > 0 and hasattr(m, "lora_alpha")
    }
    with open(os.path.join(checkpoint_dir, "lora_meta.json"), "w") as f:
        json.dump(lora_meta, f)

    logger.info(f"Saved full checkpoint at {checkpoint_dir} (epoch={epoch}, step={completed_steps})")


def load_full_state(
    checkpoint_dir: str,
    model,
    optimizer,
    lr_scheduler,
    pruning_manager,
    dataloader_generator: Optional[torch.Generator] = None,
) -> dict:
    """Restore model/optimizer/scheduler/RNG/pruning state from ``checkpoint_dir``. Returns
    {completed_steps, start_epoch, steps_trained_in_current_epoch} (split via num_update_steps_per_epoch)."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_dir}")

    _register_safe_globals_once()

    # LoRA params (strict=False — non-LoRA params are loaded from the base model).
    lora_state = torch.load(
        os.path.join(checkpoint_dir, "lora_state.pt"), map_location="cpu", weights_only=True
    )
    missing_keys, unexpected_keys = model.load_state_dict(lora_state, strict=False)
    if unexpected_keys:
        logger.warning(f"resume: unexpected keys in lora_state.pt: {unexpected_keys[:5]}")

    # Restore per-layer lora_alpha + scaling (float attrs, NOT in state_dict; a
    # fresh build defaults to alpha=2*r → wrong scaling for pruning-resized layers).
    meta_path = os.path.join(checkpoint_dir, "lora_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            lora_meta = json.load(f)
        restored = 0
        for name, m in model.named_modules():
            if name in lora_meta and hasattr(m, "lora_A") and getattr(m, "r", 0) > 0:
                m.lora_alpha = lora_meta[name]
                m.scaling = m.lora_alpha / m.r
                restored += 1
        logger.info(f"resume: restored lora_alpha/scaling for {restored} LoRA layers")

    # Surface missing LoRA keys (excluding r=0 layers, whose lora_A/B are
    # legitimately absent); base-model keys load separately via from_pretrained.
    expected_missing = set()
    if pruning_manager is not None and getattr(pruning_manager, "current_r_config", None):
        for layer, r in pruning_manager.current_r_config.items():
            if r == 0:
                expected_missing.add(f"{layer}.lora_A")
                expected_missing.add(f"{layer}.lora_B")
    lora_missing = [
        k for k in missing_keys
        if "lora_" in k and k not in expected_missing
    ]
    if lora_missing:
        logger.warning(
            f"resume: unexpected LoRA keys missing from ckpt: {lora_missing[:5]} "
            f"(total {len(lora_missing)}) — ckpt may be corrupted"
        )

    optimizer.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "optimizer.pt"), map_location="cpu", weights_only=True
    ))
    lr_scheduler.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, "scheduler.pt"), map_location="cpu", weights_only=True
    ))

    rng = torch.load(os.path.join(checkpoint_dir, "rng_state.pt"), map_location="cpu",
                     weights_only=True)
    _restore_rng_state(rng)

    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "r") as f:
        trainer = json.load(f)
    completed_steps = int(trainer["completed_steps"])
    epoch = int(trainer["epoch"])

    # HF Trainer-style mid-epoch resume.
    num_steps_per_epoch = trainer.get("num_update_steps_per_epoch")
    if num_steps_per_epoch is not None and int(num_steps_per_epoch) > 0:
        num_steps_per_epoch = int(num_steps_per_epoch)
        start_epoch = completed_steps // num_steps_per_epoch
        steps_trained_in_current_epoch = completed_steps % num_steps_per_epoch
    else:
        start_epoch = epoch + 1
        steps_trained_in_current_epoch = 0

    if pruning_manager is not None:
        load_pruning_state_json(
            pruning_manager, os.path.join(checkpoint_dir, "pruning_state.json")
        )

    # DataLoader generator state: ``generator_state`` re-draws the resumed epoch's
    # perm; ``generator_live_state`` is returned for the caller to restore after.
    dl_rng_path = os.path.join(checkpoint_dir, "dataloader_rng.pt")
    dataloader_generator_live_state = None
    if dataloader_generator is not None and os.path.isfile(dl_rng_path):
        dl_state = torch.load(dl_rng_path, map_location="cpu", weights_only=True)
        dataloader_generator.set_state(dl_state["generator_state"])
        dataloader_generator_live_state = dl_state.get("generator_live_state")

    logger.info(
        f"Resumed full checkpoint from {checkpoint_dir} "
        f"(completed_steps={completed_steps}, start_epoch={start_epoch}, "
        f"steps_trained_in_current_epoch={steps_trained_in_current_epoch})"
    )
    return {
        "completed_steps": completed_steps,
        "start_epoch": start_epoch,
        "steps_trained_in_current_epoch": steps_trained_in_current_epoch,
        "dataloader_generator_live_state": dataloader_generator_live_state,
    }
