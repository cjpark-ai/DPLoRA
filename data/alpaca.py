"""Alpaca preprocessing (Stanford convention): ``labels`` mask prompt with -100
(train on response only); no dedup/filter, ``truncation=True`` only; collator pads later."""
import logging

logger = logging.getLogger(__name__)


# Prompt templates from the Stanford Alpaca convention.
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)

ALPACA_PROMPT_NO_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def preprocess_dataset(tokenizer, raw_datasets, max_length,
                       max_train_samples=None, max_eval_samples=None,
                       seed=42, logger=None,
                       eval_split_ratio=0.0):
    """Tokenize Alpaca examples → ``(input_ids, attention_mask, labels)``; returns
    ``(train, eval)`` (eval=None unless a validation split or ``eval_split_ratio>0``)."""

    def preprocess_function(examples):
        """Tokenize source and full text separately, mask prompt by source length.
        Separate tokenization avoids a BPE off-by-one at the boundary."""
        required_keys = ['instruction', 'input', 'output']
        missing = [k for k in required_keys if k not in examples]
        if missing:
            raise ValueError(f"Alpaca dataset missing required keys: {missing}")

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for instruction, input_text, output in zip(
            examples['instruction'],
            examples['input'],
            examples['output'],
        ):
            if input_text and input_text.strip():
                source = ALPACA_PROMPT_TEMPLATE.format(
                    instruction=instruction, input=input_text)
            else:
                source = ALPACA_PROMPT_NO_INPUT_TEMPLATE.format(instruction=instruction)
            target = f"{output}{tokenizer.eos_token}"
            full_text = source + target

            full_tokenized = tokenizer(
                full_text, max_length=max_length,
                truncation=True, return_tensors=None,
                add_special_tokens=True,
            )
            source_tokenized = tokenizer(source, add_special_tokens=True)
            source_len = min(len(source_tokenized["input_ids"]), max_length)

            input_ids = list(full_tokenized["input_ids"])
            attention_mask = list(full_tokenized["attention_mask"])

            labels = list(input_ids)
            labels[:source_len] = [-100] * source_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names if "train" in raw_datasets else raw_datasets.column_names,
        desc="Preprocessing Alpaca dataset",
    )

    train_dataset = processed_datasets["train"]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))

    # eval_dataset: use only the caller-specified split_ratio; 0.0 means no eval.
    eval_dataset = processed_datasets.get("validation", processed_datasets.get("test"))
    if eval_dataset is None and eval_split_ratio > 0:
        split_datasets = train_dataset.train_test_split(
            test_size=eval_split_ratio, seed=seed
        )
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]

    if eval_dataset is not None and max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    return train_dataset, eval_dataset
