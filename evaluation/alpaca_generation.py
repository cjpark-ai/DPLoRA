"""Token-weighted perplexity + sample generations for Alpaca eval."""
import logging
import math

import torch

logger = logging.getLogger(__name__)


def evaluate_alpaca_generation(model, eval_dataloader, tokenizer, accelerator,
                               max_new_tokens=128):
    """Token-weighted perplexity (``exp(NLL/N_tokens)``) + a few sample generations.
    Token-weighting matches the paper (vs a batch-level mean over uneven token counts)."""
    model.eval()
    total_loss_x_tokens = 0.0
    total_valid_tokens = 0
    generated_texts = []

    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            # HF causal LM returns loss=None when `labels` is absent; raise
            # instead of a cryptic AttributeError on loss.detach() below.
            if loss is None:
                raise RuntimeError(
                    "evaluate_alpaca_generation: outputs.loss is None — "
                    "batch likely missing 'labels' key. Check the collator. "
                    f"Batch keys: {list(batch.keys())}"
                )
            # Loss is on shifted labels, so count valid tokens on the shifted
            # mask too (safe even if position 0 is not masked).
            valid_tokens = (batch["labels"][..., 1:] != -100).sum()
            # gather_for_metrics drops Accelerate's padded duplicates
            # (no-op in single-process mode).
            gathered_loss = accelerator.gather_for_metrics(
                loss.detach() * valid_tokens.float()
            )
            gathered_tokens = accelerator.gather_for_metrics(valid_tokens)
            total_loss_x_tokens += gathered_loss.sum().item()
            total_valid_tokens += gathered_tokens.sum().item()

            # Per-sample prompt boundary from labels mask (-100=prompt/pad). Generate
            # on main process only; no RNG save/restore (keeps determinism.py contract).
            if accelerator.is_main_process and len(generated_texts) < 10:
                input_ids = batch["input_ids"]
                labels_t = batch["labels"]
                response_start = (labels_t != -100).int().argmax(dim=1)

                n_to_take = min(batch["input_ids"].shape[0], 10 - len(generated_texts))
                for i in range(n_to_take):
                    sample_prompt_end = response_start[i].item()
                    # all-(-100) labels = fully truncated sample; skip.
                    if sample_prompt_end == 0:
                        continue

                    sample_prompt_ids = input_ids[i:i+1, :sample_prompt_end]
                    sample_generated = model.generate(
                        sample_prompt_ids,
                        # prompt slice has no padding, so attention_mask is all-ones.
                        attention_mask=torch.ones_like(sample_prompt_ids),
                        max_new_tokens=max_new_tokens,
                        # greedy decoding (deterministic; matches AlpacaEval).
                        do_sample=False,
                        # unset sampling params so they don't override greedy
                        # (silences the do_sample=False warning; no effect on argmax).
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    prompt_len_tokens = sample_prompt_ids.shape[1]
                    generated_only = sample_generated[0, prompt_len_tokens:]
                    generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)
                    generated_texts.append(generated_text)

    avg_loss = total_loss_x_tokens / max(total_valid_tokens, 1)
    # math.exp(>~710) overflows; re-raise with diagnostics so a diverged run is
    # visible and aborts.
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError as e:
        raise RuntimeError(
            f"Perplexity computation overflowed (training diverged): "
            f"avg_loss={avg_loss:.6f}, valid_tokens={total_valid_tokens}. "
            f"math.exp limit ≈ 710. Inspect the loss curve / lr / data."
        ) from e

    # all ranks share the dataset, so gather() would over-count; use
    # Accelerate's total_dataset_length, with len() as fallback.
    n_total = getattr(eval_dataloader, "total_dataset_length",
                      len(eval_dataloader.dataset))

    metrics = {
        "perplexity": perplexity,
        "eval_loss": avg_loss,
        "num_samples": int(n_total),
        "num_valid_tokens": int(total_valid_tokens),
    }

    if generated_texts:
        logger.info("Sample generations for quality assessment:")
        for i, gen in enumerate(generated_texts[:3]):
            logger.info(f"Sample {i+1}: {gen[:200]}...")

    return metrics
