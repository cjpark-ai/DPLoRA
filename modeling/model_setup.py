import logging
from functools import reduce
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import loralib as lora

logger = logging.getLogger(__name__)


class DPLoRALinear(lora.Linear):
    """loralib.Linear with a mixed-precision-safe forward (HF PEFT cast): adapter
    branch runs in adapter dtype, casts back to base. __init__/reset_parameters/merge inherited."""

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            torch_result_dtype = result.dtype
            ad = self.lora_A.dtype
            delta = (self.lora_dropout(x).to(ad)
                     @ self.lora_A.transpose(0, 1)
                     @ self.lora_B.transpose(0, 1)) * self.scaling
            # Accumulate in adapter dtype, then cast back to base dtype once
            # (HF PEFT mixed precision; no-op when the dtypes match).
            result = result + delta
            return result.to(torch_result_dtype)
        return F.linear(x, T(self.weight), bias=self.bias)


def replace_linear_with_lora(
    model: nn.Module,
    r_config: Dict[str, int],
    alpha_config: Optional[Dict[str, int]] = None,
    dropout: float = 0.0,
    merge_weights: bool = False
) -> Tuple[nn.Module, Dict[str, Dict[str, Any]]]:
    """Replace targeted ``nn.Linear`` layers with LoRA per ``r_config``. ``alpha_config``
    defaults to ``2*r``; ``r=0`` layers recorded but unwrapped. Returns (model, replaced_layers)."""
    logger.info(f"Replacing linear layers with LoRA layers using provided r configuration...")

    # Warn if r_config names a layer absent from the model (else it's silently skipped).
    existing_linear_names = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    missing_layers = sorted(set(r_config.keys()) - existing_linear_names)
    if missing_layers:
        head = missing_layers[:5]
        suffix = f" (+{len(missing_layers) - 5} more)" if len(missing_layers) > 5 else ""
        logger.warning(
            f"r_config contains {len(missing_layers)} layer(s) NOT found in model "
            f"as nn.Linear. They will be silently skipped: {head}{suffix}"
        )

    # Fail-fast on tied weights: LoRA-wrapping clones the weight and would
    # silently untie lm_head <-> embed_tokens (Llama-3, Qwen2.5).
    _name_to_module = dict(model.named_modules())
    _ptr_to_names: Dict[int, list] = {}
    for _n, _m in model.named_modules():
        _w = getattr(_m, 'weight', None)
        if isinstance(_w, torch.Tensor):
            _ptr_to_names.setdefault(_w.data_ptr(), []).append(_n)
    _tied_violations = []
    for _name in r_config:
        _m = _name_to_module.get(_name)
        if _m is None or not hasattr(_m, 'weight'):
            continue
        if not isinstance(_m.weight, torch.Tensor):
            continue
        _group = _ptr_to_names.get(_m.weight.data_ptr(), [_name])
        if len(_group) > 1:
            _tied_violations.append((_name, _group))
    if _tied_violations:
        _msg = "; ".join(f"{n} <-> {grp}" for n, grp in _tied_violations)
        raise ValueError(
            f"r_config contains tied-weight layers (LoRA-wrap would silently "
            f"break the weight tying): {_msg}. Either remove from r_config or "
            f"untie via model.config.tie_word_embeddings=False before this call."
        )

    replaced_layers = {}

    for name, module in list(model.named_modules()):
        if name in r_config and isinstance(module, nn.Linear):
            r = r_config[name]
            alpha = alpha_config.get(name, 2 * r) if alpha_config else 2 * r

            if r <= 0:
                # r=0: keep the original layer, just record it (no LoRA).
                logger.info(f"SKIPPING LoRA for {name} (r = {r}): Original layer will be preserved")
                replaced_layers[name] = {
                    'r': 0,
                    'alpha': 0,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'applied': False,
                    'trainable_params': 0,
                    'message': 'LoRA NOT applied (r=0)'
                }
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        logger.debug(f"  - Parameter {param_name} in r=0 layer will not be trained")
                continue

            logger.info(f"Replacing {name} with LoRA (r={r}, alpha={alpha})")

            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            weight_data = module.weight.data.clone()
            bias_data = module.bias.data.clone() if has_bias else None

            lora_layer = DPLoRALinear(
                in_features=in_features,
                out_features=out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=has_bias,
                merge_weights=merge_weights
            )

            # Align adapter device + dtype with the base layer (HF PEFT standard).
            lora_layer = lora_layer.to(device=weight_data.device, dtype=weight_data.dtype)

            lora_layer.weight.data = weight_data
            if has_bias:
                lora_layer.bias.data = bias_data

            # Trainable adapters in fp32, base in its loaded dtype (bf16/fp32).
            lora_layer.lora_A.data = lora_layer.lora_A.data.float()
            lora_layer.lora_B.data = lora_layer.lora_B.data.float()

            names = name.split('.')
            parent_module = reduce(getattr, names[:-1], model)
            setattr(parent_module, names[-1], lora_layer)

            # Record the replacement (same keys as the r=0 branch).
            replaced_layers[name] = {
                'r': r,
                'alpha': alpha,
                'in_features': in_features,
                'out_features': out_features,
                'applied': True,
                'trainable_params': r * (in_features + out_features),
                'message': f'LoRA applied (r={r}, alpha={alpha})'
            }

    # Single cache flush after all replacements (avoid N GPU syncs in the loop).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    applied_count = sum(1 for layer in replaced_layers.values() if layer['applied'])
    zero_r_count = sum(1 for layer in replaced_layers.values() if layer['r'] == 0)
    logger.info(
        f"LoRA Layer Summary: applied={applied_count}, r=0 skipped={zero_r_count}"
    )

    return model, replaced_layers


def prepare_model_for_glue(
    base_model_name: str,
    r_config: Dict[str, int],
    num_labels: int,
    dropout: float = 0.1,
):
    """Prepare a model with optimal LoRA for GLUE; returns ``(model, replaced_layers)``."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels
    )

    model, replaced_layers = replace_linear_with_lora(
        model, r_config, alpha_config=None, dropout=dropout, merge_weights=False
    )

    # Safe-by-default freeze; idempotent, so callers re-marking is a no-op.
    lora.mark_only_lora_as_trainable(model)
    # Stash initial r_config for later reference (used in pruning).
    model.initial_r_config = r_config.copy()

    return model, replaced_layers


def prepare_model_for_alpaca(
    base_model_name: str,
    r_config: Dict[str, int],
    dropout: float = 0.1,
    tokenizer=None,
):
    """Prepare a model with optimal LoRA for the Alpaca generation task. ``tokenizer``
    is unused (kept for backward compatibility). Returns ``(model, replaced_layers)``."""
    _ = tokenizer  # silence unused-arg linters

    if "llama" in base_model_name.lower() or "qwen" in base_model_name.lower():
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading LLaMA/Qwen model for Alpaca task: {base_model_name}")

        # bf16 base (LLaMA-3/Qwen); adapters upcast to fp32 later (HF PEFT mixed precision).
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        )
    else:
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading model for generation task: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name)

    model, replaced_layers = replace_linear_with_lora(
        model, r_config, alpha_config=None, dropout=dropout, merge_weights=False
    )

    # Safe-by-default freeze; idempotent, so callers re-marking is a no-op.
    lora.mark_only_lora_as_trainable(model)
    # Stash initial r_config for later reference (used in pruning).
    model.initial_r_config = r_config.copy()

    return model, replaced_layers
