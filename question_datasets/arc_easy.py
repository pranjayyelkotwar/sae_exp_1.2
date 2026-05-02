from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from question_datasets.base import BaseQuestionDataset


class ARCEasyDataset(BaseQuestionDataset):
    dataset_name = "allenai/ai2_arc:ARC-Easy"
    difficulty_label = "easy"
    default_split = "train"

    def load_hf_dataset(self, split: str) -> HFDataset:
        return load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)

    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        choice_labels = record["choices"]["label"]
        choice_text = record["choices"]["text"]
        ordered_choices = [
            text for _, text in sorted(zip(choice_labels, choice_text, strict=True), key=lambda item: item[0])
        ]
        answer_key = record.get("answerKey")
        return {
            "prompt_text": self.build_prompt(record["question"], ordered_choices),
            "question_text": record["question"],
            "difficulty_label": self.difficulty_label,
            "source_dataset": self.dataset_name,
            "source_split": self.split,
            "source_id": record.get("id", str(idx)),
            "subject": "science",
            "question_type": "multiple_choice",
            "choices": ordered_choices,
            "gold_answer": answer_key,
            "has_image": False,
        }
