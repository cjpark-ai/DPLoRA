"""Deterministic-environment helpers. Call ``set_deterministic_environment(seed)`` once
per entry point (after torch import); it also sets CUBLAS_WORKSPACE_CONFIG=:4096:8."""
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic_environment(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # respect a user-set CUBLAS_WORKSPACE_CONFIG instead of overwriting.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            logger.warning(f"Could not enable deterministic algorithms: {e}")

    logger.info(f"Deterministic environment set with seed {seed}")


def seed_worker(worker_id: int):
    """``DataLoader.worker_init_fn`` per-worker seeder: PyTorch auto-seeds each worker's
    torch RNG but not numpy/random; mirror ``torch.initial_seed()`` into both (official)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
