"""Bezier-curve pruning schedules for progressive LoRA pruning (smooth budget reduction)."""

import logging
import math
from typing import List, Dict, Tuple, Optional

import numpy as np

from .. import pruning_config

logger = logging.getLogger(__name__)


def calculate_bezier_coefficients(control_points: List[float]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (binomial coefficients, control points, degree) for the Bezier curve."""
    if not all(0 <= p <= 1 for p in control_points):
        raise ValueError("All control points must be between 0 and 1")

    points = np.array(control_points)
    n = len(points) - 1

    coeffs = np.zeros(n + 1)
    for i in range(n + 1):
        coeffs[i] = math.comb(n, i)

    return coeffs, points, n


def evaluate_bezier_curve(t: float, coeffs: np.ndarray, points: np.ndarray, n: int) -> float:
    """Evaluate the Bezier curve at parameter ``t`` in [0, 1]."""
    if not 0 <= t <= 1:
        raise ValueError("Parameter t must be between 0 and 1")

    result = 0.0
    for i in range(n + 1):
        result += coeffs[i] * points[i] * (t ** i) * ((1 - t) ** (n - i))

    return result


def get_pruning_schedule(
    initial_budget: int,
    target_reduction: float,
    num_pruning_steps: int,
    total_training_steps: int,
    warmup_steps: int = 0
) -> Dict[str, List[int]]:
    """Generate a Bezier-curve pruning schedule (smooth budget reduction). Raises
    ``ValueError`` on invalid input (NaN/inf reduction, non-positive budget/steps, bad warmup)."""
    # NaN/inf check (NaN comparisons silently pass <=0 / >=1 guards).
    if math.isnan(target_reduction) or target_reduction <= 0 or target_reduction >= 1:
        raise ValueError(
            f"Target reduction must be a finite value in (0, 1), got {target_reduction}"
        )

    if num_pruning_steps < 1:
        raise ValueError(f"num_pruning_steps must be >= 1, got {num_pruning_steps}")

    # Zero/negative would force all r=0.
    if initial_budget <= 0:
        raise ValueError(f"initial_budget must be > 0, got {initial_budget}")

    if total_training_steps <= 0:
        raise ValueError(f"total_training_steps must be > 0, got {total_training_steps}")

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    # Warmup must end strictly before training (otherwise interval <= 0).
    if total_training_steps <= warmup_steps:
        raise ValueError(
            f"total_training_steps ({total_training_steps}) must be > "
            f"warmup_steps ({warmup_steps})"
        )

    target_budget = initial_budget * (1 - target_reduction)

    # Get Bezier coefficients. Defensive copy so an in-place mutation of the
    # module-global list cannot alias into this call.
    control_points = list(pruning_config.BEZIER_CONTROL_POINTS)
    if len(control_points) < 2:
        # Fallback to paper-spec cubic Bezier (P = [0.0, 0.2, 0.8, 1.0]).
        control_points = [0.0, 0.2, 0.8, 1.0]

    # Reaches target_budget only if the Bezier is anchored at R(0)=0, R(N)=1 (Paper
    # Eq.13) and monotone; coeff check only validates p in [0,1], so enforce both here.
    if control_points[0] != 0.0 or control_points[-1] != 1.0:
        raise ValueError(
            f"BEZIER_CONTROL_POINTS must be anchored at 0.0 (first) and 1.0 "
            f"(last) so the final pruning step reaches target_budget; got "
            f"first={control_points[0]}, last={control_points[-1]}."
        )
    if any(control_points[i] > control_points[i + 1]
           for i in range(len(control_points) - 1)):
        raise ValueError(
            f"BEZIER_CONTROL_POINTS must be non-decreasing so the budget "
            f"schedule is monotonically non-increasing; got {control_points}."
        )

    coeffs, points, degree = calculate_bezier_coefficients(control_points)

    step_budgets = []
    step_reduction_rates = []
    step_triggers = []

    # Use warmup_steps as T_delay (warm-up period already consumed)
    # This implements: T_trigger(t) = T_delay + t * (T_total - T_delay) / (N + 1)
    T_delay = warmup_steps
    remaining_steps = total_training_steps - T_delay

    interval = remaining_steps / (num_pruning_steps + 1)

    # Fail-fast on degenerate configs: interval < 1 collides pruning triggers
    # (int() truncation), breaking the intended Bezier spacing.
    if interval < 1:
        raise ValueError(
            f"interval ({interval:.3f}) < 1: pruning step triggers would "
            f"collide. Increase (total_training_steps - warmup_steps) or "
            f"decrease num_pruning_steps so that "
            f"(total_training_steps - warmup_steps) > num_pruning_steps + 1."
        )

    for i in range(num_pruning_steps):
        # Calculate step trigger where t = i+1 (1-indexed step number)
        # Paper Eq.14: T_trigger(tau) = T_delay + tau * (T_total - T_delay) / (N + 1)
        step_trigger = int(T_delay + (i + 1) * interval)
        step_triggers.append(step_trigger)

        t = (i + 1) / num_pruning_steps  # Parameter for Bezier curve (0 to 1)
        reduction_rate = evaluate_bezier_curve(t, coeffs, points, degree)
        current_reduction = target_reduction * reduction_rate
        current_budget = initial_budget * (1 - current_reduction)  # Paper Eq.12: B_step(tau) = B_initial * (1 - R(tau))

        current_budget = int(current_budget)

        step_budgets.append(current_budget)
        step_reduction_rates.append(current_reduction)

    schedule = {
        "initial_budget": initial_budget,
        "target_budget": int(target_budget),
        "step_budgets": step_budgets,
        "step_reduction_rates": step_reduction_rates,
        "step_triggers": step_triggers,
        "T_delay": T_delay,
        "interval": interval
    }

    logger.info(f"Generated pruning schedule: initial={initial_budget:,} → target={int(target_budget):,}")
    logger.info(f"  T_delay (warm-up period): {T_delay} steps")
    logger.info(f"  Uniform pruning interval: {interval:.2f} steps")
    for i, (budget, rate, trigger) in enumerate(zip(step_budgets, step_reduction_rates, step_triggers)):
        logger.info(f"  Step {i+1}: budget={budget:,} (reduction={rate:.2%}) at training step {trigger}")

    return schedule


class PruningScheduler:
    """Stateful pruning-schedule machine, driven per OPTIMIZER step (should_prune→prune+
    start_recovery+advance_pruning_step; update_recovery every step). Violations fail silently."""

    def __init__(
        self,
        initial_budget: int,
        target_reduction: float,
        num_pruning_steps: int,
        total_training_steps: int,
        warmup_steps: int = 0,
        recovery_steps: Optional[int] = None,
        extended_recovery_steps: Optional[int] = None,
    ):
        self.schedule = get_pruning_schedule(
            initial_budget=initial_budget,
            target_reduction=target_reduction,
            num_pruning_steps=num_pruning_steps,
            total_training_steps=total_training_steps,
            warmup_steps=warmup_steps
        )

        # Caller must supply both values (no ``pruning_config`` fallback).
        if recovery_steps is None:
            raise ValueError("recovery_steps is required (no pruning_config fallback)")
        if extended_recovery_steps is None:
            raise ValueError("extended_recovery_steps is required (no pruning_config fallback)")
        self._default_recovery_steps = recovery_steps
        self._extended_recovery_steps = extended_recovery_steps

        # A recovery window overrunning the next pruning trigger keeps should_prune()
        # False → silent under-reduction, so fail fast. (extended_recovery path not guarded.)
        interval = self.schedule["interval"]
        if recovery_steps >= interval:
            raise ValueError(
                f"recovery_steps ({recovery_steps}) >= pruning interval "
                f"({interval:.2f}); recovery would overrun the next trigger so "
                f"pruning steps get delayed/skipped and target_budget may never "
                f"be reached. Reduce recovery_steps or increase "
                f"(total_training_steps - warmup_steps) / (num_pruning_steps + 1)."
            )

        self.current_step_idx = -1  # No pruning done yet
        self.in_recovery_mode = False
        self.recovery_step_counter = 0
        self.recovery_steps = self._default_recovery_steps

    def should_prune(self, training_step: int) -> bool:
        """Whether pruning should fire at ``training_step`` (False while in recovery)."""
        if self.in_recovery_mode:
            return False

        next_step_idx = self.current_step_idx + 1
        if next_step_idx < len(self.schedule["step_triggers"]):
            if training_step >= self.schedule["step_triggers"][next_step_idx]:
                return True

        return False

    def start_recovery(self, extended: bool = False) -> None:
        """Enter recovery mode (extended window after a rollback)."""
        self.in_recovery_mode = True
        self.recovery_step_counter = 0
        self.recovery_steps = self._extended_recovery_steps if extended else self._default_recovery_steps

    def update_recovery(self) -> None:
        """Update recovery step counter and check if recovery is complete."""
        if self.in_recovery_mode:
            self.recovery_step_counter += 1
            if self.recovery_step_counter >= self.recovery_steps:
                self.in_recovery_mode = False

    def get_next_pruning_budget(self) -> int:
        next_step_idx = self.current_step_idx + 1
        if next_step_idx < len(self.schedule["step_budgets"]):
            return self.schedule["step_budgets"][next_step_idx]
        return self.schedule["target_budget"]

    def advance_pruning_step(self) -> None:
        self.current_step_idx += 1

    def get_current_step_info(self) -> Dict[str, float]:
        if self.current_step_idx < 0:
            return {
                "step_idx": -1,
                "budget": self.schedule["initial_budget"],
                "reduction_rate": 0.0,
                "total_steps": len(self.schedule["step_budgets"])
            }

        # Clamp current_step_idx to prevent IndexError when advance_pruning_step is
        # called > num_pruning_steps times (mirrors get_next_pruning_budget).
        n_steps = len(self.schedule["step_budgets"])
        idx = min(self.current_step_idx, n_steps - 1)
        return {
            "step_idx": idx,
            "budget": self.schedule["step_budgets"][idx],
            "reduction_rate": self.schedule["step_reduction_rates"][idx],
            "total_steps": n_steps
        }