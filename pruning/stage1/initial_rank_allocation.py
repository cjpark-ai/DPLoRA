import gc
import logging
import os
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

from .. import pruning_config
from ..pruning_utils import estimate_parameter_cost

logger = logging.getLogger(__name__)


def measure_layer_importance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layers: List[str],
    device: torch.device,
    num_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Stage 1 layer importance (Paper Eq.1): I_l^init = E_x[mean((grad theta_l L)^2)],
    per-sample backward (diagonal Fisher), averaged over num_batches. Call BEFORE
    training (does not back up existing grads)."""
    if num_batches is None:
        num_batches = pruning_config.IMPORTANCE_NUM_BATCHES

    use_memory_efficient_mode = os.getenv('USE_MEMORY_EFFICIENT_IMPORTANCE', 'false').lower() == 'true'
    chunk_size = int(os.getenv('IMPORTANCE_CHUNK_SIZE', '8'))

    if use_memory_efficient_mode:
        logger.info(f"Measuring layer importance with memory-efficient chunking (chunk_size={chunk_size})...")
    else:
        logger.info("Measuring layer importance...")

    model.to(device)
    model.eval()

    gradient_checkpointing_was_enabled = False
    if use_memory_efficient_mode and hasattr(model, 'gradient_checkpointing_enable'):
        if hasattr(model.config, 'gradient_checkpointing') and not model.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            gradient_checkpointing_was_enabled = True
            logger.info("Gradient checkpointing enabled for memory efficiency")

    # Accumulate per-sample (grad**2).mean() on-device as a tensor; defer .item()
    # to the end of the loop to avoid a GPU→CPU sync per sample-layer.
    importances_acc = {
        name: torch.zeros((), device=device, dtype=torch.float32)
        for name in target_layers
    }
    sample_counts = {name: 0 for name in target_layers}

    linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in target_layers:
            linear_layers[name] = module

    # Fail-fast: Stage-1 importance needs trainable target weights; frozen layers
    # would silently contribute importance=0 (biasing them to the smallest r).
    frozen = [n for n, m in linear_layers.items() if not m.weight.requires_grad]
    if frozen:
        raise ValueError(
            f"Stage-1 importance requires trainable target layer weights; "
            f"{len(frozen)}/{len(linear_layers)} layer(s) have "
            f"requires_grad=False: {frozen[:5]}"
        )

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        if use_memory_efficient_mode and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch_size = batch['input_ids'].size(0) if 'input_ids' in batch else list(batch.values())[0].size(0)

        # Process sample by sample to ensure abs → mean order
        for sample_idx in range(batch_size):
            sample_batch = {k: v[sample_idx:sample_idx+1] for k, v in batch.items()}

            model.zero_grad(set_to_none=True)
            outputs = model(**sample_batch)

            # Use loss if available, otherwise create a pseudo-loss.
            loss = None
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            elif hasattr(outputs, 'logits') and outputs.logits is not None:
                if 'labels' in sample_batch:
                    if outputs.logits.dim() > 1 and outputs.logits.size(-1) > 1:
                        loss = torch.nn.functional.cross_entropy(outputs.logits, sample_batch['labels'])
                    else:
                        loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), sample_batch['labels'].float())
                else:
                    loss = torch.mean(outputs.logits ** 2)

            if loss is None:
                logger.warning(f"Batch {batch_idx}, Sample {sample_idx}: Unable to compute loss, skipping")
                continue

            loss.backward()

            # Importance: mean of squared grads (Fisher Information approximation);
            # accumulate as tensor, single .item() at end of fn.
            for name, layer in linear_layers.items():
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    importances_acc[name] = importances_acc[name] + (layer.weight.grad ** 2).mean()
                    sample_counts[name] += 1

            if use_memory_efficient_mode and torch.cuda.is_available() and sample_idx % 4 == 0:
                torch.cuda.empty_cache()

    # materialize tensor accumulator → Python float dict with one .item() per layer.
    importances = {}
    for name in target_layers:
        if sample_counts[name] > 0:
            importances[name] = (importances_acc[name] / sample_counts[name]).item()
        else:
            importances[name] = 0.0

    # NaN/Inf replacement (NaN → small sentinel, Inf → large sentinel).
    # Raises RuntimeError if every value is NaN/Inf.
    from ..pruning_utils import _apply_nan_inf_replacement
    _apply_nan_inf_replacement(importances)

    # Optional normalization (off by default; shared with Stage 2 via the
    # same pruning_config.NORMALIZE_IMPORTANCE flag).
    if pruning_config.NORMALIZE_IMPORTANCE and importances:
        max_importance = max(importances.values())
        if max_importance > 0:
            for name in importances:
                importances[name] /= max_importance

    if pruning_config.LOG_IMPORTANCE_SCORES:
        logger.info("Layer importance scores (top 10):")
        for name, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {name}: {importance:.6f}")

    if gradient_checkpointing_was_enabled and hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled after importance calculation")

    return importances


def estimate_performance_gain(
    r_value: int,
    layer_importance: float,
    layer_size: Tuple[int, int],
) -> float:
    """Estimate the LoRA performance gain for ``r_value`` on a layer (Paper Eq.2: G_{l,r} = I_l^init * (1 - e^{-r/M_l}), M_l = min(d_in, d_out))."""
    in_features, out_features = layer_size
    max_rank = min(in_features, out_features)
    normalized_rank = r_value / max_rank

    gain = layer_importance * (1 - np.exp(-normalized_rank))
    return gain


def prepare_optimization_data(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layers: List[str],
    r_values: List[int],
    device: torch.device,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, float]]:
    """Build the ILP inputs: return (gains, costs, importances), each keyed by
    layer_name (gains/costs are per-r lists; importances reused by the caller)."""
    logger.info(f"Preparing optimization data for {len(target_layers)} layers and {len(r_values)} r values...")

    importances = measure_layer_importance(model, dataloader, target_layers, device)

    layer_sizes = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in target_layers:
            layer_sizes[name] = (module.in_features, module.out_features)

    gains = {layer_name: [] for layer_name in target_layers}
    costs = {layer_name: [] for layer_name in target_layers}

    for layer_name in target_layers:
        if layer_name not in layer_sizes:
            logger.warning(f"Layer {layer_name} not found in model, skipping...")
            continue

        importance = importances.get(layer_name, 0)
        layer_size = layer_sizes[layer_name]

        for r in r_values:
            gain = estimate_performance_gain(r, importance, layer_size)
            cost = estimate_parameter_cost(r, layer_size)

            gains[layer_name].append(gain)
            costs[layer_name].append(cost)

    return gains, costs, importances


def optimize_r_values(
    gains: Dict[str, List[float]],
    costs: Dict[str, List[float]],
    r_values: List[int],
    budget: float,
    time_limit: int = None,
    seed: int = 42
) -> Dict[str, int]:
    """Optimize r per layer via ILP (Paper Eq.3): maximize sum_l sum_r G_{l,r} x_{l,r}
    s.t. one r per layer and sum C_{l,r} x_{l,r} <= budget, x in {0,1}. Returns layer→r."""
    try:
        import pulp
    except ImportError:
        logger.error("PuLP is required for this optimization. Please install pulp.")
        logger.error("If you cannot install PuLP, consider using a simple greedy algorithm as fallback.")
        raise ImportError("PuLP optimizer is required for reliable optimization.")

    start_time = time.time()
    logger.info(f"Starting r-value optimization with budget {budget:.2f}...")

    layer_names = list(gains.keys())

    if time_limit is None:
        time_limit = pruning_config.OPTIMIZATION_TIMEOUT

    logger.info(f"Setting optimization time limit to {time_limit} seconds and seed to {seed}")

    os.environ["CBC_RANDOM_SEED"] = str(seed)
    np.random.seed(seed)

    ilp_model = pulp.LpProblem("optimal_r", pulp.LpMaximize)

    # CBC 2.10.3 solver options (PuLP-bundled). Determinism: randomSeed + threads 1
    # + zero gaps. Time limit via PULP_CBC_CMD(timeLimit=); don't dup "sec"/"timeLimit" here.
    solver_options = [
        f"randomSeed {seed}",
        "threads 1",
        "ratioGap 0.0",
        "allowableGap 0.0",
        "presolve on",
        "passPresolve 5",
        "cutoff 1e50",
        "strong 10",
        "perturbation on",
        "passC 1000",
        "cuts on",
        "passCuts 10",
        "cost off",
        "logLevel 1",
        "nodeStrategy depth",
        "integerT 1e-9",
        "primalT 1e-9",
        "dualT 1e-9",
        "autoScale on",
    ]

    variables = {}
    for layer_name in layer_names:
        for i, r in enumerate(r_values):
            variables[(layer_name, i)] = pulp.LpVariable(
                f"{layer_name}_r{r}",
                cat=pulp.LpBinary
            )

    # Constraint 1: each layer chooses exactly one r value.
    for layer_name in layer_names:
        ilp_model += (
            pulp.lpSum(variables[(layer_name, i)] for i in range(len(r_values))) == 1,
            f"one_r_{layer_name}"
        )

    # Constraint 2: total parameter cost must not exceed the budget.
    total_cost = pulp.lpSum(
        costs[layer_name][i] * variables[(layer_name, i)]
        for layer_name in layer_names
        for i in range(len(r_values))
    )
    ilp_model += (total_cost <= budget, "budget_constraint")

    # Constraint 3: aggregate rank within each layer type >= 1
    # (prevents component collapse, e.g. ALL q_proj layers becoming r=0).
    from .. import pruning_utils
    layer_types = pruning_utils.classify_layers_by_type(layer_names)

    for layer_type, type_layers in layer_types.items():
        if not type_layers:
            continue

        sum_r_expr = pulp.lpSum(
            r_values[i] * variables[(layer_name, i)]
            for layer_name in type_layers for i in range(len(r_values))
        )

        ilp_model += (
            sum_r_expr >= 1,
            f"min_rank_{layer_type}"
        )

    # Objective: Maximize total gain
    objective = pulp.lpSum(
        gains[layer_name][i] * variables[(layer_name, i)]
        for layer_name in layer_names
        for i in range(len(r_values))
    )

    ilp_model += objective

    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=time_limit,
        options=solver_options,
        keepFiles=False,
        mip=True,
        threads=1,       # Redundant but explicit single thread setting
        gapRel=0.0,
        gapAbs=0.0,
    )

    logger.info(f"Starting deterministic optimization with CBC solver (seed: {seed}, threads: 1)")

    solve_start = time.time()
    ilp_model.solve(solver)
    solve_time = time.time() - solve_start

    status = pulp.LpStatus[ilp_model.status]
    if status == 'Optimal':
        logger.info(f"Optimal solution found in {solve_time:.2f} seconds!")
    elif status == 'Not Solved':
        logger.warning(f"Time limit reached after {solve_time:.2f} seconds, returning best solution found so far.")
    else:
        # fail-fast instead of silent fallback. ILPNotSolvedError imported lazily
        # from pruning_utils to avoid a circular import.
        from ..pruning_utils import ILPNotSolvedError
        raise ILPNotSolvedError(
            f"Stage-1 ILP failed with status {status!r} after {solve_time:.2f}s "
            f"(neither 'Optimal' nor 'Not Solved')."
        )

    optimal_r = {}
    total_gain = 0
    total_cost_value = 0

    for layer_name in layer_names:
        for i, r in enumerate(r_values):
            # pulp.value returns None when CBC times out / crashes without an incumbent.
            v = pulp.value(variables[(layer_name, i)])
            if v is not None and v > 0.5:
                optimal_r[layer_name] = r_values[i]
                total_gain += gains[layer_name][i]
                total_cost_value += costs[layer_name][i]
                break  # one r per layer

    # fail-fast if any layer has no assignment. Stage-1 is the initial allocation
    # — no prior r_config to fall back on, so surface the failure.
    missing = [n for n in layer_names if n not in optimal_r]
    if missing:
        from ..pruning_utils import ILPNotSolvedError
        raise ILPNotSolvedError(
            f"Stage-1 ILP returned no r-value for {len(missing)}/{len(layer_names)} "
            f"layers (status={status!r}, solve_time={solve_time:.2f}s). "
            f"First missing: {missing[:3]}. This typically indicates CBC timed "
            f"out or crashed without an incumbent."
        )

    elapsed_time = time.time() - start_time
    logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
    logger.info(f"Optimal solution with total gain: {total_gain:.4f}")
    logger.info(f"Total cost: {total_cost_value:.2f} / {budget:.2f} ({100 * total_cost_value / budget:.1f}%)")

    for layer_name in sorted(optimal_r.keys()):
        r = optimal_r[layer_name]
        i = r_values.index(r)
        logger.info(f"{layer_name}: r={r}, gain={gains[layer_name][i]:.4f}, cost={costs[layer_name][i]:.2f}")

    return optimal_r


def get_optimal_r_config(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    r_values: List[int],
    target_layers: List[str],
    budget: float,
    device: torch.device,
    seed: int = 42
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Run Stage-1 importance + ILP; return (optimal_r, optimization_results)."""
    logger.info(f"Computing optimal r configuration with budget {budget}...")
    logger.info(f"Considering r values: {r_values}")
    logger.info(f"Optimizing {len(target_layers)} layers with seed {seed}")

    gains, costs, importances = prepare_optimization_data(
        model, dataloader, target_layers, r_values, device
    )

    optimal_r = optimize_r_values(gains, costs, r_values, budget, seed=seed)

    optimization_results = {
        "gains": gains,
        "costs": costs,
        "r_values": r_values,
        "budget": budget,
        "seed": seed,
        "importances": importances,
    }

    return optimal_r, optimization_results