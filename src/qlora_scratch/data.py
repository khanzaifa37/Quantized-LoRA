from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def build_oasst1_splits(
    output_dir: str | Path,
    *,
    train_ratio: float = 0.85,
    seed: int = 42,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> DatasetDict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("OpenAssistant/oasst1", split="train")

    messages = {}
    children = defaultdict(list)
    for row in raw:
        messages[row["message_id"]] = row
        if row["parent_id"]:
            children[row["parent_id"]].append(row["message_id"])

    texts = []
    for row in raw:
        if row["parent_id"] is not None:
            continue
        for child_id in children.get(row["message_id"], []):
            child = messages[child_id]
            if child["role"] != "assistant":
                continue
            texts.append(
                template.format(
                    instruction=row["text"].strip(),
                    response=child["text"].strip(),
                )
            )
            break

    combined = Dataset.from_dict({"text": texts}).shuffle(seed=seed)
    split = combined.train_test_split(test_size=1.0 - train_ratio, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)

    dataset = DatasetDict(
        {
            "train": split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    for split_name, ds in dataset.items():
        ds.to_json(str(output_dir / f"{split_name}.json"))

    metadata = {
        "dataset_name": "OpenAssistant/oasst1",
        "prompt_template": template,
        "num_examples": {name: len(ds) for name, ds in dataset.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return dataset


def load_json_splits(data_dir: str | Path) -> DatasetDict:
    data_dir = Path(data_dir)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.json"),
            "validation": str(data_dir / "validation.json"),
            "test": str(data_dir / "test.json"),
        },
    )
    return DatasetDict(dataset)
