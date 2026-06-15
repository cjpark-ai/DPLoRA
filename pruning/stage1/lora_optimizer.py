import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .initial_rank_allocation import get_optimal_r_config
from utils.logging_utils import log_optimal_r_config

logger = logging.getLogger(__name__)


class LoRAOptimizer:
    """Optimizer for finding the optimal LoRA rank configuration."""

    def __init__(
        self,
        model: nn.Module,
        r_values: List[int],
        budget: float,
        device: torch.device,
        output_dir: str,
        seed: int = 42
    ):
        if not r_values:
            raise ValueError("r_values cannot be empty — ILP needs at least one candidate r")
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        if not output_dir:
            raise ValueError("output_dir cannot be empty or None")

        self.model = model
        self.r_values = r_values
        self.budget = budget
        self.device = device
        self.output_dir = output_dir
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initialized LoRA Optimizer with seed {seed}")

    def find_target_layers(self, layer_patterns: List[str] = None) -> List[str]:
        """Find linear layers matching ``layer_patterns`` (None = auto-detect by model type)."""
        if layer_patterns is None:
            # Single-pass model type detection (one named_modules() walk).
            model_type_names = {type(m).__name__ for m in self.model.modules()}
            is_qwen = any('Qwen2' in t for t in model_type_names)
            is_llama = any('Llama' in t for t in model_type_names)
            is_bart = any('Bart' in t for t in model_type_names)
            is_t5 = any('T5' in t for t in model_type_names)

            # Target-module suffixes matched via ``name.endswith`` (PEFT dot-prefix);
            # the leading dot enforces a module boundary (excludes classifier/pooler.dense).
            if is_qwen:
                layer_patterns = [".q_proj", ".k_proj", ".v_proj", ".o_proj",
                                  ".gate_proj", ".up_proj", ".down_proj"]
                logger.info("Detected Qwen model, using Qwen layer patterns")
            elif is_llama:
                layer_patterns = [".q_proj", ".k_proj", ".v_proj", ".o_proj",
                                  ".gate_proj", ".up_proj", ".down_proj"]
                logger.info("Detected LLaMA model, using LLaMA layer patterns")
            elif is_bart:
                layer_patterns = [".q_proj", ".k_proj", ".v_proj", ".out_proj",
                                  ".fc1", ".fc2"]
                logger.info("Detected BART model, using BART layer patterns")
            elif is_t5:
                layer_patterns = [".q", ".k", ".v", ".o",
                                  ".wi", ".wi_0", ".wi_1", ".wo"]
                logger.info("Detected T5 model, using T5 layer patterns")
            else:
                # BERT/RoBERTa — dot-prefix suffix forms (excludes classifier head).
                layer_patterns = [".query", ".key", ".value",
                                  ".attention.output.dense",
                                  ".intermediate.dense",
                                  ".output.dense"]
                logger.info("Using BERT/RoBERTa layer patterns")

        target_layers = []

        linear_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_count += 1
                if linear_count <= 5:
                    logger.debug(f"Found linear layer: {name}")

                # HF PEFT canonical (modules_to_save): classifier / pooler are
                # kept as full modules, not LoRA-wrapped.
                if 'classifier' in name or 'pooler' in name:
                    continue

                for pattern in layer_patterns:
                    if name.endswith(pattern):
                        target_layers.append(name)
                        break

        logger.info(f"Total linear layers: {linear_count}")
        logger.info(f"Found {len(target_layers)} target layers for optimization")
        for i, layer in enumerate(sorted(target_layers)):
            logger.info(f"  {i+1}. {layer}")

        # Fail fast on an empty target list — otherwise the downstream ILP fails silently.
        if not target_layers:
            raise ValueError(
                f"No target layers matched any of the {len(layer_patterns)} patterns. "
                f"Total Linear layers in model: {linear_count}. "
                f"Patterns: {layer_patterns}"
            )

        return target_layers

    def optimize(
        self,
        dataloader: DataLoader,
        target_layers: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Find the optimal r config (layer_name -> r); target_layers None = auto-detect."""
        if target_layers is None:
            target_layers = self.find_target_layers()  # raises ValueError if empty

        # Guard against an explicitly empty list passed by the caller.
        if not target_layers:
            raise ValueError(
                "target_layers is empty — cannot optimize r values for 0 layers. "
                "Either pass non-empty target_layers or rely on auto-detection (None)."
            )

        logger.info(f"Optimizing with seed {self.seed}")

        logger.info("Starting optimization process...")
        optimal_r, optimization_results = get_optimal_r_config(
            model=self.model,
            dataloader=dataloader,
            r_values=self.r_values,
            target_layers=target_layers,
            budget=self.budget,
            device=self.device,
            seed=self.seed,
        )

        # seed already in optimization_results (get_optimal_r_config); persisted to
        # reproducible_r_config.json / optimization_results.json by log_optimal_r_config.
        log_optimal_r_config(
            logger=logger,
            optimal_r=optimal_r,
            optimization_results=optimization_results,
            output_dir=self.output_dir
        )

        return optimal_r
