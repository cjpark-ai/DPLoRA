"""Configuration constants for progressive LoRA pruning: only constants NOT exposed
via the training-script argparse (other hyperparameters are passed through ``args.*``)."""

# Stage 1+2 importance (Paper Eq.1+Eq.6): I_l^init = E_x[mean((grad_theta_l L(x))^2)],
# averaged over IMPORTANCE_NUM_BATCHES mini-batches (≤IMPORTANCE_MAX_SAMPLES); both stages share.
IMPORTANCE_MAX_SAMPLES = 500
IMPORTANCE_NUM_BATCHES = 5         # Number of mini-batches used (Paper: "five mini-batches")

PERFORMANCE_DROP_THRESHOLD = 0.03  # Acceptable performance drop (3%)

NORMALIZE_IMPORTANCE = False       # Whether to normalize importance scores (paper omits — use raw)
OPTIMIZATION_TIMEOUT = 600         # Timeout for ILP optimization (seconds)

BEZIER_CONTROL_POINTS = [0.0, 0.2, 0.8, 1.0]

LOG_IMPORTANCE_SCORES = True
PLOT_PRUNING_TRAJECTORY = True