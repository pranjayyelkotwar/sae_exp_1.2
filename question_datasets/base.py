import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

class BaseQuestionDataset(Dataset, ABC):
    dataset_name: str
    difficulty_label: str
    default_split: str

    def __init__(
        self,
        tokenizer: Any,
        max_token_length: int,
        split: str | None = None,
        add_bos_token: bool = False,
        include_choices: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.add_bos_token = add_bos_token
        self.include_choices = include_choices
        self.split = split or self.default_split

        logging.info("Loading %s (%s split)...", self.dataset_name, self.split)
        dataset = self.load_hf_dataset(split=self.split)
        self.dataset = dataset.filter(self._keep_example)
        logging.info("Loaded %s examples from %s", len(self.dataset), self.dataset_name)

    @abstractmethod
    def load_hf_dataset(self, split: str) -> HFDataset:
        raise NotImplementedError

    @abstractmethod
    def normalize_record(self, record: dict[str, Any], idx: int) -> dict[str, Any]:
        raise NotImplementedError

    def _keep_example(self, record: dict[str, Any]) -> bool:
        return not self._has_visual_component(record)

    @staticmethod
    def _has_visual_component(record: dict[str, Any]) -> bool:
        image_like_keys = (
            "image",
            "images",
            "figure",
            "figures",
            "image_path",
            "image_paths",
            "image_url",
            "image_urls",
        )
        for key in image_like_keys:
            value = record.get(key)
            if BaseQuestionDataset._value_contains_payload(value):
                return True
        return False

    @staticmethod
    def _value_contains_payload(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, bytes):
            return len(value) > 0
        if isinstance(value, dict):
            return any(BaseQuestionDataset._value_contains_payload(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(BaseQuestionDataset._value_contains_payload(item) for item in value)
        return True

    @staticmethod
    def _format_choices(choices: list[str] | None) -> str:
        if not choices:
            return ""
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{labels[idx]}. {choice}" for idx, choice in enumerate(choices))

    def build_prompt(self, question_text: str, choices: list[str] | None = None) -> str:
        prompt_lines = ["Answer the following question.", "", f"Question: {question_text.strip()}"]
        if self.include_choices and choices:
            prompt_lines.extend(["Choices:", self._format_choices(choices)])
        return "\n".join(prompt_lines)

    def encode_prompt(self, prompt_text: str) -> list[int]:
        tokens = self.tokenizer.encode(prompt_text, bos=self.add_bos_token, eos=False)
        return tokens[: self.max_token_length]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[list[int], int, int, dict[str, Any]]:
        record = self.normalize_record(self.dataset[idx], idx)
        prompt_tokens = self.encode_prompt(record["prompt_text"])
        seq_len = len(prompt_tokens)
        metadata = {**record, "token_count": seq_len}
        return prompt_tokens, idx, seq_len, metadata

    def collate_fn(
        self,
        batch: list[tuple[list[int], int, int, dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        max_seq_len = max(seq_len for _, _, seq_len, _ in batch)
        padded_prompt_tokens = [
            prompt_tokens + [self.tokenizer.pad_id] * (max_seq_len - seq_len)
            for prompt_tokens, _, seq_len, _ in batch
        ]
        collated_batch = (
            torch.tensor(padded_prompt_tokens, dtype=torch.long),
            torch.tensor([idx for _, idx, _, _ in batch]),
            torch.tensor([seq_len for _, _, seq_len, _ in batch]),
            [metadata for _, _, _, metadata in batch],
        )
        return collated_batch
