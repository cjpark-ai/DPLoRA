"""Stage-2 progressive pruning: per-step ILP rank re-allocation with recovery/rollback during training."""

import json
import logging
import os
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import pruning_config
from .. import pruning_utils
from ..pruning_utils import estimate_parameter_cost, ILPNotSolvedError

logger = logging.getLogger(__name__)


def optimize_pruning_step(
    r_config: Dict[str, int],
    layer_importances: Dict[str, float],
    layer_sizes: Dict[str, Tuple[int, int]],
    available_r_values: List[int],
    step_budget: int,
    momentum_penalty: float = 1.0,
    stable_layer_bonus: float = 0.05,
    time_limit: int = pruning_config.OPTIMIZATION_TIMEOUT,
    seed: int = 42
) -> Dict[str, int]:
    """Solve a single-step pruning ILP and return the optimized r config (layer_name -> r)."""
    logger.info(f"Starting pruning optimization with budget {step_budget:,} parameters and seed {seed}")

    try:
        import pulp
    except ImportError:
        logger.error("PuLP is required for this optimization. Please install pulp.")
        raise ImportError("PuLP optimizer is required for pruning optimization.")

    start_time = time.time()
    layer_names = list(r_config.keys())

    valid_r_values = {layer: [r for r in available_r_values if r <= r_config[layer]]
                     for layer in layer_names}

    logger.info("Formulating ILP problem...")
    ilp_model = pulp.LpProblem("lora_pruning", pulp.LpMinimize)

    os.environ["CBC_RANDOM_SEED"] = str(seed)
    np.random.seed(seed)

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
    for layer in layer_names:
        for r in valid_r_values[layer]:
            variables[(layer, r)] = pulp.LpVariable(
                f"{layer}_r{r}",
                cat=pulp.LpBinary
            )

    # Constraint 1 (Paper Eq.11): each layer chooses exactly one r value.
    for layer in layer_names:
        ilp_model += (
            pulp.lpSum(variables[(layer, r)] for r in valid_r_values[layer]) == 1,
            f"one_r_{layer}"
        )

    # Constraint 2 (Paper Eq.11): monotonic pruning r_new <= r_current, enforced via
    # the valid_r_values filter (avoids ~16 unused binary vars/layer vs explicit constraint).

    # Constraint 3 (Paper Eq.11): total parameter cost <= step budget; cost via
    # estimate_parameter_cost (Paper Eq.4), symmetric with Stage 1.
    total_params = pulp.lpSum(
        estimate_parameter_cost(r, layer_sizes[layer]) * variables[(layer, r)]
        for layer in layer_names
        for r in valid_r_values[layer]
    )
    ilp_model += (total_params <= step_budget, "budget_constraint")

    # Constraint 4 (Paper Eq.5, both stages): aggregate rank per layer type >= 1
    # (prevents component collapse, e.g. all q_proj → r=0).
    layer_types = pruning_utils.classify_layers_by_type(layer_names)

    for layer_type, type_layers in layer_types.items():
        if not type_layers:
            continue

        sum_r_expr = pulp.lpSum(
            r * variables[(layer, r)]
            for layer in type_layers
            for r in valid_r_values[layer]
        )

        ilp_model += (
            sum_r_expr >= 1,
            f"min_rank_{layer_type}"
        )

    # Objective function: Minimize estimated performance loss with momentum-based penalty
    # Paper Eq.9: J_Stage2 = sum_l sum_r (L_{l,r_new} + P_{l,r_new}) * x_{l,r_new}
    objective = pulp.LpAffineExpression()

    for layer in layer_names:
        current_r = r_config[layer]
        # 0.0 fallback means "no learning signal" (matches Stage 1).
        importance = layer_importances.get(layer, 0.0)

        for r in valid_r_values[layer]:
            loss = pruning_utils.estimate_performance_loss(
                current_r=current_r,
                new_r=r,
                importance=importance,
            )

            # Paper Eq.10: P_{l,r_new} = gamma * I_l^(t) * |r_new - r_current| - delta * I_l^(t) * 1[r_new = r_current]; gamma = momentum_penalty, delta = stable_layer_bonus
            r_diff = abs(r - current_r)
            momentum_term = r_diff * momentum_penalty * importance
            if r == current_r:
                momentum_term -= stable_layer_bonus * importance
            loss += momentum_term

            if r_diff > 0 and logger.level <= logging.DEBUG:
                logger.debug(f"Layer {layer}: r={current_r}→{r}, momentum penalty: {momentum_term:.6f}")

            objective += loss * variables[(layer, r)]

    ilp_model += objective

    logger.info(f"Starting deterministic optimization with CBC solver (seed: {seed}, threads: 1)")

    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=time_limit,
        options=solver_options,
        keepFiles=False,
        mip=True,
        threads=1,       # redundant with options but explicit
        gapRel=0.0,
        gapAbs=0.0
    )

    solve_start = time.time()
    ilp_model.solve(solver)
    solve_time = time.time() - solve_start

    status = pulp.LpStatus[ilp_model.status]
    if status == 'Optimal':
        logger.info(f"Optimal solution found in {solve_time:.2f} seconds!")
    elif status == 'Not Solved':
        logger.warning(f"Time limit reached after {solve_time:.2f} seconds, using best solution found so far")
    else:
        # fail-fast instead of silent fallback.
        raise ILPNotSolvedError(
            f"Pruning ILP failed with status {status!r} after {solve_time:.2f}s "
            f"(neither 'Optimal' nor 'Not Solved')."
        )

    new_r_config = {}
    for layer in layer_names:
        for r in valid_r_values[layer]:
            # pulp.value returns None when CBC times out without finding a
            # feasible solution (the variable was never assigned).
            v = pulp.value(variables[(layer, r)])
            if v is not None and v > 0.5:
                new_r_config[layer] = r
                break

        # Fallback if no solution found for a layer (covers the None-value case).
        if layer not in new_r_config:
            logger.warning(f"No r value selected for {layer}, keeping current r={r_config[layer]}")
            new_r_config[layer] = r_config[layer]

    initial_params = pruning_utils.calculate_total_parameters(r_config, layer_sizes)
    final_params = pruning_utils.calculate_total_parameters(new_r_config, layer_sizes)
    reduction = initial_params - final_params
    reduction_percentage = (reduction / initial_params * 100) if initial_params > 0 else 0

    logger.info(f"Optimization completed in {time.time() - start_time:.2f}s")
    logger.info(f"Parameter reduction: {reduction:,} parameters ({reduction_percentage:.2f}%)")
    logger.info(f"Final parameter count: {final_params:,} parameters")

    # Analyze the influence of momentum penalty if applicable
    stable_count = sum(1 for layer in layer_names if new_r_config[layer] == r_config.get(layer, 0))
    changed_count = len(layer_names) - stable_count
    if changed_count > 0:
        avg_change = sum(abs(new_r_config[layer] - r_config.get(layer, 0)) for layer in layer_names) / len(layer_names)
        logger.info(f"Stability: {stable_count}/{len(layer_names)} layers unchanged, avg |Δr|: {avg_change:.2f}")
    else:
        logger.info("Stability: All layers remained unchanged")

    pruned_layers = sum(1 for layer in layer_names if new_r_config[layer] < r_config[layer])
    unchanged_layers = sum(1 for layer in layer_names if new_r_config[layer] == r_config[layer])
    increased_layers = sum(1 for layer in layer_names if new_r_config[layer] > r_config[layer])

    logger.info(f"R-value changes: {pruned_layers} decreased, {unchanged_layers} unchanged, {increased_layers} increased")

    pruned_info = []
    for layer in layer_names:
        if new_r_config[layer] < r_config[layer]:
            param_diff = (r_config[layer] - new_r_config[layer]) * sum(layer_sizes[layer])
            pruned_info.append((layer, r_config[layer], new_r_config[layer], param_diff))

    if pruned_info:
        logger.info(f"Top pruned layers (by parameter reduction):")
        for layer, old_r, new_r, param_diff in sorted(pruned_info, key=lambda x: x[3], reverse=True)[:5]:
            logger.info(f"  {layer}: r={old_r} → {new_r} (-{param_diff:,} parameters)")

    return new_r_config


class ProgressivePruningManager:
    """Drive Stage-2 progressive pruning (ILP re-allocation + recovery + optional rollback) during training."""

    def __init__(
        self,
        model: nn.Module,
        initial_r_config: Dict[str, int],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        target_reduction: float,
        num_pruning_steps: int,
        total_training_steps: int,
        device: torch.device,
        output_dir: str = "pruning_results",
        seed: int = 42,
        warmup_steps: int = 0,
        enable_rollback: bool = False,
        importance_ema_decay: Optional[float] = None,
        momentum_penalty_weight: Optional[float] = None,
        stable_layer_bonus: Optional[float] = None,
        recovery_steps: Optional[int] = None,
        extended_recovery_steps: Optional[int] = None,
        available_r_values: Optional[List[int]] = None,
        importance_dataloader: Optional[DataLoader] = None,
    ):
        """Initialize from caller-supplied hyperparameters (no pruning_config fallback)."""
        self.model = model
        self.initial_r_config = initial_r_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = output_dir
        self.seed = seed
        self.enable_rollback = enable_rollback
        self._pre_pruning_opt_snapshot = None

        # Store all pruning hyperparameters from the caller (argparse-bound script
        # args); no fallback to ``pruning_config`` module globals.
        if importance_ema_decay is None:
            raise ValueError("importance_ema_decay is required (no pruning_config fallback)")
        if momentum_penalty_weight is None:
            raise ValueError("momentum_penalty_weight is required (no pruning_config fallback)")
        if stable_layer_bonus is None:
            raise ValueError("stable_layer_bonus is required (no pruning_config fallback)")
        if recovery_steps is None:
            raise ValueError("recovery_steps is required (no pruning_config fallback)")
        if extended_recovery_steps is None:
            raise ValueError("extended_recovery_steps is required (no pruning_config fallback)")
        if available_r_values is None:
            raise ValueError("available_r_values is required (no pruning_config fallback)")
        # Stage-2 importance on a fixed-N deterministic train subset (Paper Eq.6);
        # caller builds it with shuffle=False + per_device_eval_batch_size.
        if importance_dataloader is None:
            raise ValueError(
                "importance_dataloader is required (Stage-2 importance "
                "signal source — fixed-N deterministic subset of train data)"
            )
        self.importance_ema_decay = importance_ema_decay
        self.momentum_penalty_weight = momentum_penalty_weight
        self.stable_layer_bonus = stable_layer_bonus
        self.recovery_steps = recovery_steps
        self.extended_recovery_steps = extended_recovery_steps
        self.available_r_values = available_r_values
        self.importance_dataloader = importance_dataloader

        # A layer whose current rank ∉ available_r_values can never be retained
        # (filtered to {r<=current_r}∩available), so stable_layer_bonus is moot. Misconfig only.
        _missing = sorted({r for r in initial_r_config.values()
                           if r not in available_r_values})
        if _missing:
            raise ValueError(
                f"initial_r_config contains rank(s) {_missing} absent from "
                f"available_r_values {sorted(available_r_values)}; those layers "
                f"cannot retain their current rank and would be force-changed. "
                f"Ensure available_r_values is a superset of the initial ranks."
            )

        logger.info(f"EMA decay factor: {self.importance_ema_decay}")
        logger.info(f"Momentum penalty weight: {self.momentum_penalty_weight}")
        logger.info(f"Stable layer bonus: {self.stable_layer_bonus}")
        logger.info(f"Recovery steps: {self.recovery_steps}")
        logger.info(f"Extended recovery steps: {self.extended_recovery_steps}")

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initializing progressive pruning manager with seed {seed}")

        self.layer_sizes = {}
        for layer_name in initial_r_config:
            self.layer_sizes[layer_name] = pruning_utils.get_layer_size(model, layer_name)

        self.initial_params = pruning_utils.calculate_total_parameters(initial_r_config, self.layer_sizes)
        logger.info(f"Initial LoRA parameters: {self.initial_params:,}")

        from .pruning_scheduler import PruningScheduler
        self.scheduler = PruningScheduler(
            initial_budget=self.initial_params,
            target_reduction=target_reduction,
            num_pruning_steps=num_pruning_steps,
            total_training_steps=total_training_steps,
            warmup_steps=warmup_steps,
            recovery_steps=self.recovery_steps,
            extended_recovery_steps=self.extended_recovery_steps,
        )

        # Pre-flight feasibility: ILP needs each layer type's sum(r)>=1 (Eq.5, fixed
        # min cost); if the smallest step budget is below that floor, fail fast here.
        _floor = pruning_utils.per_type_feasibility_floor(
            list(initial_r_config.keys()), self.layer_sizes,
            initial_r_config, self.available_r_values,
        )
        _min_budget = min(self.scheduler.schedule["step_budgets"])
        if _min_budget < _floor:
            raise ValueError(
                f"Infeasible pruning schedule: smallest step budget "
                f"({_min_budget:,}) is below the per-type feasibility floor "
                f"({_floor:,}) required by the sum(r) >= 1 constraint (Eq.5). "
                f"Lower target_reduction, add a smaller value to "
                f"available_r_values, or use a model with more layers per type."
            )

        self.current_r_config = initial_r_config.copy()
        self.current_performance = None
        self.baseline_performance = None
        self.pruning_history = []
        self.checkpoint_path = os.path.join(output_dir, "pre_pruning_checkpoint.pt")

        self.pruning_metrics = {
            'eval_flops': 0,
            'eval_macs': 0,
            'pruned_params': 0,
            'model_size_reduction': 0,
            'accuracy_per_gflops': 0,
            'accuracy_per_gmacs': 0,
            'eval_steps_per_second': 0
        }

        self.prev_importances = None  # For EMA of layer importances

        # Last layer-wise change record from modify_lora_layers (for optimizer resync)
        self.last_changes = {}

    def initialize_baseline_performance(self) -> None:
        self.baseline_performance = pruning_utils.evaluate_model(
            model=self.model,
            eval_dataloader=self.eval_dataloader,
            device=self.device
        )
        self.current_performance = self.baseline_performance

        logger.info(f"Baseline performance: {self.baseline_performance:.4f}")

        self.pruning_history.append({
            'step': 0,
            'r_config': self.current_r_config.copy(),
            'param_count': self.initial_params,
            'performance': self.baseline_performance,
            'status': 'initial'
        })

        logger.info("=" * 80)
        logger.info("INITIAL MODEL PARAMETERS BREAKDOWN")
        logger.info("=" * 80)

        layer_types = pruning_utils.classify_layers_by_type(list(self.initial_r_config.keys()))

        for layer_type, layers in layer_types.items():
            type_params = 0
            for layer in layers:
                r = self.initial_r_config.get(layer, 0)
                if r > 0 and layer in self.layer_sizes:
                    in_features, out_features = self.layer_sizes[layer]
                    layer_params = r * (in_features + out_features)
                    type_params += layer_params

            type_percentage = (type_params / self.initial_params * 100) if self.initial_params > 0 else 0
            logger.info(f"{layer_type} layers: {len(layers)} layers, {type_params:,} parameters ({type_percentage:.2f}%)")

        logger.info(f"Total LoRA parameters: {self.initial_params:,}")
        logger.info("=" * 80)

    def should_prune(self, training_step: int) -> bool:
        """Whether pruning should fire at ``training_step``."""
        return self.scheduler.should_prune(training_step)

    def execute_pruning_step(self, training_step: int, optimizer=None) -> bool:
        """Run one pruning step (importance → ILP re-alloc → modify layers → recovery);
        True on success. With ``optimizer``, snapshots+resyncs its state around pruning."""

        # Always use INFO level for critical stages
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            logger.info(f"\n{'='*80}\nEXECUTING PRUNING STEP AT TRAINING STEP {training_step}\n{'='*80}")

            # Save checkpoint for potential rollback (skip if rollback disabled to
            # avoid ~100MB * N disk I/O on large models).
            if self.enable_rollback:
                pruning_utils.save_model_checkpoint(self.model, self.checkpoint_path)
                if optimizer is not None:
                    self._pre_pruning_opt_snapshot = pruning_utils._snapshot_optimizer_state(
                        optimizer, self.model, clone=True
                    )

            step_budget = self.scheduler.get_next_pruning_budget()
            step_info = self.scheduler.get_current_step_info()
            next_step_idx = step_info["step_idx"] + 1

            logger.info(f"Pruning step {next_step_idx+1}/{step_info['total_steps']} - "
                       f"Target budget: {step_budget:,} parameters")

            prev_r_config = self.current_r_config.copy()

            # measure_layer_importance returns (importances, raw_ema); store raw_ema as
            # prev_importances (raw-scale EMA). Uses fixed-N eval_batch dataloader (Eq.6).
            layer_importances, raw_ema = pruning_utils.measure_layer_importance(
                model=self.model,
                dataloader=self.importance_dataloader,
                r_config=self.current_r_config,
                device=self.device,
                prev_importances=self.prev_importances,
                ema_decay=self.importance_ema_decay
            )

            self.prev_importances = raw_ema

            new_r_config = optimize_pruning_step(
                r_config=self.current_r_config,
                layer_importances=layer_importances,
                layer_sizes=self.layer_sizes,
                available_r_values=self.available_r_values,
                step_budget=step_budget,
                momentum_penalty=self.momentum_penalty_weight,
                stable_layer_bonus=self.stable_layer_bonus,
                time_limit=pruning_config.OPTIMIZATION_TIMEOUT,
                seed=self.seed
            )

            logger.info("=" * 60)
            logger.info("LAYER-WISE R-VALUE CHANGES PLANNED")
            logger.info("=" * 60)

            increased = 0
            decreased = 0
            unchanged = 0

            layer_types = pruning_utils.classify_layers_by_type(list(self.current_r_config.keys()))

            for layer_type, layers in layer_types.items():
                logger.info(f"\n[{layer_type.upper()} LAYERS]")
                for layer_name in sorted(layers):
                    old_r = self.current_r_config.get(layer_name, 0)
                    new_r = new_r_config.get(layer_name, 0)

                    if old_r > new_r:
                        change_symbol = "↓"
                        decreased += 1
                        change_info = f"PRUNED ({old_r-new_r} reduction)"
                    elif old_r < new_r:
                        change_symbol = "↑"
                        increased += 1
                        change_info = f"INCREASED ({new_r-old_r} addition)"
                    else:
                        change_symbol = "="
                        unchanged += 1
                        change_info = "UNCHANGED"

                    in_features, out_features = self.layer_sizes.get(layer_name, (0, 0))
                    old_params = old_r * (in_features + out_features) if old_r > 0 else 0
                    new_params = new_r * (in_features + out_features) if new_r > 0 else 0

                    logger.info(f"  {layer_name}: r={old_r} {change_symbol} {new_r} - {change_info}")
                    if old_params > 0 or new_params > 0:
                        logger.info(f"    Parameters: {old_params:,} → {new_params:,} ({new_params-old_params:+,})")

            logger.info("\nSUMMARY:")
            logger.info(f"  Decreased r-values: {decreased} layers")
            logger.info(f"  Unchanged r-values: {unchanged} layers")
            logger.info(f"  Increased r-values: {increased} layers")
            logger.info("=" * 60)

            # Snapshot optimizer state BEFORE pruning to re-sync replaced Parameters
            # (modify_lora_layers reassigns lora_A/B, orphaning them from the optimizer).
            optimizer_snapshot = None
            if optimizer is not None:
                optimizer_snapshot = pruning_utils._snapshot_optimizer_state(
                    optimizer, self.model
                )

            self.model, changes = pruning_utils.modify_lora_layers(
                model=self.model,
                new_r_config=new_r_config
            )
            self.last_changes = changes  # expose for caller / debugging

            # Re-sync optimizer with replaced parameters (preserves Adam state
            # via slicing/padding for rank-changed layers, verbatim for unchanged).
            if optimizer is not None and changes:
                pruning_utils.resync_optimizer_with_model(
                    optimizer, self.model, changes, optimizer_snapshot
                )

            if not changes:
                logger.info("No changes made to model, skipping validation")
                # Record a 'no_change' step so current_step_idx stays consistent with
                # len(pruning_history) (else the counter drifts ahead of the rows).
                self.pruning_history.append({
                    'step': next_step_idx + 1,
                    'r_config': self.current_r_config.copy(),
                    'param_count': pruning_utils.calculate_total_parameters(
                        self.current_r_config, self.layer_sizes),
                    'performance': self.current_performance,
                    'status': 'no_change',
                })
                # Advance scheduler anyway
                self.scheduler.advance_pruning_step()
                return True

            pruning_validation, validation_summary = pruning_utils.validate_pruning_configuration(
                model=self.model,
                prev_r_config=prev_r_config,
                new_r_config=new_r_config,
                layer_sizes=self.layer_sizes,
                logger=logger
            )

            if validation_summary['mismatch_layers'] > 0:
                logger.error(f"CRITICAL: Found {validation_summary['mismatch_layers']} layer(s) with r-value mismatches!")
                for layer_name, details in pruning_validation.items():
                    if not details['is_match']:
                        logger.error(f"Mismatch in layer {layer_name}: expected r={details['expected_r']}, "
                                    f"got r={details['actual_r']}")

            self.pruning_metrics['pruned_params'] = validation_summary['total_reduction']
            self.pruning_metrics['model_size_reduction'] = validation_summary['reduction_percentage']

            self.scheduler.start_recovery()

            current_params = pruning_utils.calculate_total_parameters(new_r_config, self.layer_sizes)

            self.current_r_config = new_r_config

            self.pruning_history.append({
                'step': next_step_idx + 1,
                'r_config': self.current_r_config.copy(),
                'param_count': current_params,
                'performance': None,  # Will be updated after recovery
                'status': 'pruned',
                'pruning_details': validation_summary
            })

            self.scheduler.advance_pruning_step()

            return True

        finally:
            logger.setLevel(original_level)

    def update_recovery(self, optimizer=None) -> None:
        """Advance recovery; on completion, validate (and possibly roll back) the model.
        ``optimizer`` is forwarded so the rollback path can re-sync Adam state."""
        self.scheduler.update_recovery()

        # If recovery just completed, validate model
        if not self.scheduler.in_recovery_mode and self.pruning_history[-1]['status'] == 'pruned':
            self._validate_after_recovery(optimizer=optimizer)

    def _validate_after_recovery(self, optimizer=None) -> None:
        """Evaluate after recovery; roll back if the performance drop exceeds threshold.
        With ``optimizer`` + rollback firing, re-syncs its (m,v) state to the rolled-back shape."""

        # Force INFO level for visibility of critical logs
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            start_time = time.time()
            performance, is_acceptable = pruning_utils.validate_pruning(
                model=self.model,
                eval_dataloader=self.eval_dataloader,
                baseline_performance=self.baseline_performance,
                threshold=pruning_config.PERFORMANCE_DROP_THRESHOLD,
                device=self.device
            )
            eval_time = time.time() - start_time

            num_eval_steps = len(self.eval_dataloader)
            steps_per_second = num_eval_steps / eval_time if eval_time > 0 else 0

            self.current_performance = performance

            self.pruning_history[-1]['performance'] = performance
            self.pruning_history[-1]['eval_time'] = eval_time
            self.pruning_history[-1]['eval_steps_per_second'] = steps_per_second

            self.pruning_metrics['eval_steps_per_second'] = steps_per_second

            logger.info("=" * 60)
            logger.info(f"PRUNING STEP {len(self.pruning_history)-1} VALIDATION RESULTS")
            logger.info("=" * 60)

            current_params = self.pruning_history[-1]['param_count']
            param_reduction = self.initial_params - current_params
            param_reduction_pct = (param_reduction / self.initial_params * 100) if self.initial_params > 0 else 0

            logger.info(f"Parameters: {self.initial_params:,} → {current_params:,} ({current_params-self.initial_params:+,}, {param_reduction_pct:.2f}% reduction)")
            logger.info(f"Performance: {self.baseline_performance:.4f} → {performance:.4f} ({(performance-self.baseline_performance):+.4f})")
            logger.info(f"Evaluation speed: {steps_per_second:.2f} steps/second")

            if is_acceptable or not self.enable_rollback:
                success_msg = "SUCCESS - Performance is within acceptable threshold"
                if not is_acceptable and not self.enable_rollback:
                    success_msg = "ACCEPTED (rollback disabled) - Performance drop exceeded threshold but rollback is disabled"

                self.pruning_history[-1]['status'] = 'success'
                logger.info(f"VERDICT: {success_msg}")

                if self.enable_rollback:
                    pruning_utils.save_model_checkpoint(self.model, self.checkpoint_path)
                    if optimizer is not None:
                        self._pre_pruning_opt_snapshot = pruning_utils._snapshot_optimizer_state(
                            optimizer, self.model, clone=True
                        )
            else:
                logger.warning(f"VERDICT: FAILED - Performance drop exceeds threshold, rolling back")

                # Rollback (off by default; unused in paper): modify_lora_layers reshapes
                # to pre-prune rank BEFORE load_model_checkpoint, else the ckpt is truncated.
                prev_r_config = self.pruning_history[-2]['r_config'].copy()

                # Step 1: expand lora_A/B back to pre-pruning shape
                self.model, rollback_changes = pruning_utils.modify_lora_layers(
                    model=self.model,
                    new_r_config=prev_r_config
                )

                # Step 2: overwrite weight values from the pre-pruning checkpoint
                self.model = pruning_utils.load_model_checkpoint(
                    self.model,
                    self.checkpoint_path,
                    allow_size_mismatch=True  # shapes already match after step 1
                )

                # Restore EXACT pre-pruning optimizer state (snapshot taken at
                # checkpoint-save time: weights + momentum both at the pre-pruning point).
                if optimizer is not None and self._pre_pruning_opt_snapshot is not None:
                    pruning_utils._restore_optimizer_state(
                        optimizer, self.model, self._pre_pruning_opt_snapshot
                    )

                self.current_r_config = prev_r_config
                self.current_performance = self.pruning_history[-2]['performance']

                # Update history entry to reflect the rolled-back state (not the
                # post-pruning state).
                self.pruning_history[-1]['status'] = 'rollback'
                self.pruning_history[-1]['r_config'] = self.current_r_config.copy()
                self.pruning_history[-1]['param_count'] = pruning_utils.calculate_total_parameters(
                    self.current_r_config, self.layer_sizes
                )
                self.pruning_history[-1]['performance'] = self.current_performance

                self.scheduler.start_recovery(extended=True)

            logger.info("=" * 60)

        finally:
            logger.setLevel(original_level)

    def get_progress_info(self) -> Dict[str, Any]:
        """Return a snapshot of pruning progress (params, reduction, performance, step counts)."""
        current_params = pruning_utils.calculate_total_parameters(
            self.current_r_config, self.layer_sizes)
        reduction = (self.initial_params - current_params) / self.initial_params

        return {
            "initial_params": self.initial_params,
            "current_params": current_params,
            "reduction": reduction,
            "target_reduction": self.scheduler.schedule["step_reduction_rates"][-1],
            "current_performance": self.current_performance,
            "baseline_performance": self.baseline_performance,
            "steps_completed": self.scheduler.current_step_idx + 1,
            "total_steps": len(self.scheduler.schedule["step_budgets"]),
            "in_recovery": self.scheduler.in_recovery_mode,
            "pruning_metrics": self.pruning_metrics
        }

    def finalize(self) -> Tuple[nn.Module, Dict[str, int], float]:
        """Finalize pruning, write reports/JSON, and return (model, final_r_config, achieved_reduction)."""
        current_params = pruning_utils.calculate_total_parameters(
            self.current_r_config, self.layer_sizes)
        achieved_reduction = (self.initial_params - current_params) / self.initial_params

        self.pruning_metrics['pruned_params'] = self.initial_params - current_params
        self.pruning_metrics['model_size_reduction'] = achieved_reduction * 100

        logger.info(f"\n{'='*80}\nPROGRESSIVE PRUNING COMPLETED\n{'='*80}")
        logger.info(f"Initial parameters = {self.initial_params:,}")
        logger.info(f"Final parameters = {current_params:,}")
        logger.info(f"Achieved reduction = {achieved_reduction:.2%}")
        logger.info(f"Final performance = {self.current_performance:.4f} (baseline: {self.baseline_performance:.4f})")

        if self.pruning_metrics['eval_flops'] > 0:
            logger.info(f"Performance per GFLOPs = {self.pruning_metrics['accuracy_per_gflops']:.6f}")

        if self.pruning_metrics['eval_macs'] > 0:
            logger.info(f"Performance per GMACs = {self.pruning_metrics['accuracy_per_gmacs']:.6f}")

        logger.info(f"Evaluation speed = {self.pruning_metrics['eval_steps_per_second']:.2f} steps/second")

        if pruning_config.PLOT_PRUNING_TRAJECTORY and len(self.pruning_history) > 1:
            pruning_utils.plot_pruning_results(
                pruning_history=self.pruning_history,
                output_dir=self.output_dir
            )

        with open(os.path.join(self.output_dir, "final_r_config.json"), 'w') as f:
            json.dump(self.current_r_config, f, indent=2)

        with open(os.path.join(self.output_dir, "pruning_metrics.json"), 'w') as f:
            metrics = {
                "initial_params": self.initial_params,
                "final_params": current_params,
                "reduction": float(achieved_reduction),
                "baseline_performance": float(self.baseline_performance),
                "final_performance": float(self.current_performance),
                "pruning_metrics": self.pruning_metrics
            }
            json.dump(metrics, f, indent=2)

        return self.model, self.current_r_config, achieved_reduction