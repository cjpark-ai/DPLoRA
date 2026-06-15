"""GLUE evaluation pass: one pass feeds gathered predictions to a ``compute_metrics_fn``
closure (bit-exact to ``evaluate.load("glue", task)``); MNLI mismatched runs a 2nd time."""
import logging
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


def evaluate_glue(model, eval_dataloader, compute_metrics_fn, task_name, accelerator):
    """Run one eval pass on a GLUE dataloader. Returns a dict: eval_metric, eval_loss,
    eval_runtime, eval_steps_per_second, eval_samples_per_second, total_samples."""
    model.eval()

    eval_start_time = time.time()

    eval_loss_numer = 0.0
    eval_loss_denom = 0
    all_predictions = []
    all_references = []

    def _gather_to_list(tensor):
        # gather_for_metrics removes padded duplicates.
        gathered = accelerator.gather_for_metrics(tensor)
        return gathered.detach().cpu().numpy()

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

            bs = batch["input_ids"].size(0)
            if outputs.loss is not None:
                # sample-weighted accumulator (HF Trainer equivalent).
                eval_loss_numer += outputs.loss.detach().float().item() * bs
                eval_loss_denom += bs

            predictions = (
                outputs.logits.argmax(dim=-1)
                if task_name != "stsb"
                # squeeze(-1): avoid a 0-dim scalar on a size-1 last batch
                # (np.concatenate would fail).
                else outputs.logits.squeeze(-1)
            )

            if "labels" in batch:
                all_predictions.append(_gather_to_list(predictions))
                all_references.append(_gather_to_list(batch["labels"]))
            else:
                matched_alt = False
                for key in ("label", "target", "targets"):
                    if key in batch:
                        logger.info(f"Found alternative label key: {key}")
                        all_predictions.append(_gather_to_list(predictions))
                        all_references.append(_gather_to_list(batch[key]))
                        matched_alt = True
                        break
                if not matched_alt:
                    logger.warning(f"No label key found in batch. Keys: {list(batch.keys())}")

    eval_runtime = time.time() - eval_start_time

    if all_predictions:
        preds = np.concatenate(all_predictions, axis=0)
        refs = np.concatenate(all_references, axis=0)
        eval_metric = compute_metrics_fn(preds, refs)
    else:
        raise RuntimeError(
            f"evaluate_glue: no valid (labeled) batches collected for task "
            f"'{task_name}'. Check the eval dataloader and collator."
        )

    avg_eval_loss = (eval_loss_numer / eval_loss_denom
                      if eval_loss_denom > 0 else 0.0)

    # Accelerate standard attribute for correct total sample count.
    total_samples = getattr(eval_dataloader, "total_dataset_length",
                            len(eval_dataloader.dataset))

    eval_steps_per_second = (
        len(eval_dataloader) / eval_runtime if eval_runtime > 0 else 0.0
    )
    eval_samples_per_second = (
        total_samples / eval_runtime if eval_runtime > 0 else 0.0
    )

    return {
        "eval_metric": eval_metric,
        "eval_loss": avg_eval_loss,
        "eval_runtime": eval_runtime,
        "eval_steps_per_second": eval_steps_per_second,
        "eval_samples_per_second": eval_samples_per_second,
        "total_samples": total_samples,
    }
