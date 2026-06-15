import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _numpy_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Set up a logger writing to console (+ file if given); logs system/environment
    info once at INFO level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # avoid duplicate logs when re-setting up the same logger.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if level <= logging.INFO:
        try:
            import platform

            logger.info("=" * 80)
            logger.info("SYSTEM ENVIRONMENT INFORMATION FOR REPRODUCIBILITY")
            logger.info("=" * 80)
            logger.info(f"Python version: {platform.python_version()}")
            logger.info(f"OS: {platform.platform()}")
            logger.info(f"CPU: {platform.processor()}")

            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"NumPy version: {np.__version__}")

            try:
                import pulp
                logger.info(f"PuLP version: {pulp.__version__}")
            except (ImportError, AttributeError):
                logger.info("PuLP not available")

            logger.info("=" * 80)

        except Exception as e:
            logger.warning(f"Error logging system information: {e}")

    return logger


def log_optimal_r_config(
    logger: logging.Logger,
    optimal_r: Dict[str, int],
    optimization_results: Dict[str, Any],
    output_dir: str
):
    """Log the optimal r config and save JSON configs + r-value and cost-gain
    plots to ``output_dir``."""
    required_keys = ("r_values", "gains", "costs")
    missing = [k for k in required_keys if k not in optimization_results]
    if missing:
        raise ValueError(
            f"optimization_results missing required key(s): {missing}. "
            f"Required keys: {list(required_keys)}."
        )

    seed = optimization_results.get("seed", None)
    seed_info = f" (seed: {seed})" if seed is not None else ""
    logger.info(f"Optimal r configuration{seed_info}:")

    sorted_layers = sorted(optimal_r.keys())
    for layer_name in sorted_layers:
        r = optimal_r[layer_name]
        logger.info(f"  {layer_name}: r={r}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "optimal_r_config.json"), "w") as f:
        json.dump(optimal_r, f, indent=2, default=_numpy_safe)

    reproducible_config = {
        "optimal_r": optimal_r,
        "reproducibility": {
            "seed": optimization_results.get("seed", None),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "r_values": optimization_results.get("r_values", []),
            "budget": optimization_results.get("budget", 0)
        }
    }

    with open(os.path.join(output_dir, "reproducible_r_config.json"), "w") as f:
        json.dump(reproducible_config, f, indent=2, default=_numpy_safe)

    logger.info(f"Saved reproducible configuration to {os.path.join(output_dir, 'reproducible_r_config.json')}")

    layers = list(optimal_r.keys())
    r_values = list(optimal_r.values())

    short_layer_names = [name.split('.')[-1] if '.' in name else name for name in layers]

    layer_types = []
    for name in layers:
        if "query" in name:
            layer_types.append("query")
        elif "key" in name:
            layer_types.append("key")
        elif "value" in name:
            layer_types.append("value")
        elif "attention" in name:
            layer_types.append("attention")
        elif "intermediate" in name:
            layer_types.append("ffn")
        elif "output" in name:
            layer_types.append("output")
        else:
            layer_types.append("other")

    sorted_indices = np.argsort(layer_types)
    sorted_layers = [layers[i] for i in sorted_indices]
    sorted_r_values = [r_values[i] for i in sorted_indices]
    sorted_short_names = [short_layer_names[i] for i in sorted_indices]
    sorted_layer_types = [layer_types[i] for i in sorted_indices]

    color_map = {
        "query": "blue",
        "key": "green",
        "value": "red",
        "attention": "purple",
        "ffn": "orange",
        "output": "brown",
        "other": "gray"
    }
    colors = [color_map[t] for t in sorted_layer_types]

    fig1 = plt.figure(figsize=(15, 10))
    plt.bar(range(len(sorted_layers)), sorted_r_values, color=colors)
    plt.xticks(range(len(sorted_layers)), sorted_short_names, rotation=90)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[t], label=t)
                      for t in sorted(set(layer_types))]
    plt.legend(handles=legend_elements)

    plt.xlabel('Layer')
    plt.ylabel('Optimal r value')
    plt.title('Optimal r configuration')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimal_r_config.png"))
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 6))

    layer_gains = []
    layer_costs = []
    r_value_indices = {r: i for i, r in enumerate(optimization_results["r_values"])}

    for layer_name in sorted_layers:
        r = optimal_r[layer_name]
        if r not in r_value_indices:
            raise ValueError(
                f"optimal_r['{layer_name}']={r} is not in "
                f"optimization_results['r_values']={optimization_results['r_values']}."
            )
        r_idx = r_value_indices[r]
        gain = optimization_results["gains"][layer_name][r_idx]
        cost = optimization_results["costs"][layer_name][r_idx]
        layer_gains.append(gain)
        layer_costs.append(cost)

    plt.scatter(layer_costs, layer_gains, c=colors, alpha=0.7)

    for i, layer_name in enumerate(sorted_short_names):
        plt.annotate(layer_name, (layer_costs[i], layer_gains[i]),
                    textcoords="offset points", xytext=(0,5), ha='center')

    plt.xlabel('Cost')
    plt.ylabel('Gain')
    plt.title('Cost vs. Gain for Each Layer')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_gain_scatter.png"))
    plt.close(fig2)

    with open(os.path.join(output_dir, "optimization_results.json"), "w") as f:
        json.dump(optimization_results, f, indent=2, default=_numpy_safe)