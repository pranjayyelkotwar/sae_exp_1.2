from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from question_datasets.base import BaseQuestionDataset


class MMLUDataset(BaseQuestionDataset):
    dataset_name = "cais/mmlu"
    difficulty_label = "medium"
    default_split = "test"

    def load_hf_dataset(self, split: str) -> HFDataset:
        return load_dataset("cais/mmlu", "all", split=split)

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        answer = record.get("answer")
        if isinstance(answer, int):
            answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer]
        return {
            "prompt_text": self.build_prompt(record["question"], record.get("choices")),
            "question_text": record["question"],
            "difficulty_label": self.difficulty_label,
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": str(idx),
            "subject": record.get("subject", "unknown"),
            "question_type": "multiple_choice",
            "choices": record.get("choices"),
            "gold_answer": answer,
            "has_image": False,
        }
