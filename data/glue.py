"""GLUE preprocessing. ``task_to_keys`` maps each task to its input column(s)
(None 2nd = single-sentence); ``preprocess_dataset`` tokenizes + copies label → ``labels``."""
import logging

logger = logging.getLogger(__name__)


task_to_keys = {
    # WNLI omitted (LoRA-pruning convention; LoRA-drop / PRILoRA / LoRAPrune
    # all report results on the canonical 8 GLUE tasks below).
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


def preprocess_dataset(tokenizer, raw_datasets, task_name, max_length,
                       max_train_samples=None, max_eval_samples=None):
    """Tokenise + label-copy a GLUE task's train/validation splits. Returns
    ``(train, eval)``, or for MNLI ``(train, eval_matched, eval_mismatched)``."""
    sentence1_key, sentence2_key = task_to_keys[task_name]

    def _check_non_empty(values, key_name):
        """Warn (don't raise) on empty input strings — surfaces silently-dropped
        samples on custom data; standard GLUE splits have none."""
        empties = sum(1 for s in values
                      if isinstance(s, str) and not s.strip())
        if empties:
            logger.warning(
                f"GLUE preprocess: {empties} empty value(s) found in column "
                f"'{key_name}' (task={task_name})."
            )

    def preprocess_function(examples):
        _check_non_empty(examples[sentence1_key], sentence1_key)
        if sentence2_key is not None:
            _check_non_empty(examples[sentence2_key], sentence2_key)
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)

        if "label" not in examples:
            # Fail fast rather than silently skipping examples in the training loop.
            raise ValueError(
                f"No 'label' column found in examples. Keys: {list(examples.keys())}"
            )
        result["labels"] = examples["label"]

        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Preprocessing dataset",
    )

    train_dataset = processed_datasets["train"]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))

    if task_name == "mnli":
        eval_matched_dataset = processed_datasets["validation_matched"]
        eval_mismatched_dataset = processed_datasets["validation_mismatched"]

        if max_eval_samples is not None:
            eval_matched_dataset = eval_matched_dataset.select(
                range(min(max_eval_samples, len(eval_matched_dataset)))
            )
            eval_mismatched_dataset = eval_mismatched_dataset.select(
                range(min(max_eval_samples, len(eval_mismatched_dataset)))
            )

        return train_dataset, eval_matched_dataset, eval_mismatched_dataset

    eval_dataset = processed_datasets["validation"]
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    return train_dataset, eval_dataset


def make_compute_metrics(task_name: str):
    """GLUE compute_metrics closure, bit-exact to HF ``evaluate.load("glue", task).compute``.
    stsb adds ``corr=(pearson+spearmanr)/2`` (paper, for ckpt selection); WNLI unhandled."""
    import numpy as np
    from sklearn.metrics import matthews_corrcoef, f1_score
    from scipy.stats import pearsonr, spearmanr

    def compute(predictions, references):
        preds = np.asarray(predictions)
        refs = np.asarray(references)
        if task_name == "cola":
            return {"matthews_correlation": matthews_corrcoef(refs, preds)}
        if task_name == "stsb":
            pearson = float(pearsonr(preds, refs)[0])
            spearman = float(spearmanr(preds, refs)[0])
            return {
                "pearson": pearson,
                "spearmanr": spearman,
                "corr": (pearson + spearman) / 2.0,
            }
        if task_name in ("mrpc", "qqp"):
            return {
                "accuracy": float((preds == refs).mean()),
                "f1": float(f1_score(y_true=refs, y_pred=preds)),
            }
        # sst2, mnli (matched/mismatched), qnli, rte, wnli
        return {"accuracy": float((preds == refs).mean())}

    return compute
