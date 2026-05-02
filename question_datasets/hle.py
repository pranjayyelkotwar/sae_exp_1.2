from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from question_datasets.base import BaseQuestionDataset


class HLEDataset(BaseQuestionDataset):
    dataset_name = "cais/hle"
    difficulty_label = "hard"
    default_split = "test"

    def load_hf_dataset(self, split: str) -> HFDataset:
        return load_dataset("cais/hle", split=split)

    def _keep_example(self, record: dict[str, Any]) -> bool:
        return (
            record.get("image") == ""
            # or self._value_contains_payload(record.get("image_preview"))
            # or self._value_contains_payload(record.get("rationale_image"))
        )

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        choices = (
            record.get("choices")
            or record.get("options")
            or record.get("multiple_choice_targets")
        )
        answer = (
            record.get("answer")
            or record.get("answer_key")
            or record.get("correct_answer")
            or record.get("solution")
        )
        source_id = (
            record.get("question_id")
            or record.get("id")
            or record.get("uid")
            or str(idx)
        )
        subject = (
            record.get("subject")
            or record.get("category")
            or record.get("raw_subject")
            or record.get("field")
            or "unknown"
        )
        return {
            "prompt_text": self.build_prompt(record["question"], choices),
            "question_text": record["question"],
            "difficulty_label": self.difficulty_label,
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": source_id,
            "subject": subject,
            "question_type": record.get("answer_type") or ("multiple_choice" if choices else "short_answer"),
            "choices": choices,
            "gold_answer": answer,
            "has_image": False,
        }
