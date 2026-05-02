from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

from question_datasets.arc_easy import ARCEasyDataset
from question_datasets.base import BaseQuestionDataset
from question_datasets.hle import HLEDataset
from question_datasets.mmlu import MMLUDataset


DATASET_REGISTRY = {
    "arc_easy": ARCEasyDataset,
    "mmlu": MMLUDataset,
    "hle": HLEDataset,
}


class LimitedDataset(Dataset):
    """Wrapper to limit a dataset to a maximum number of samples."""
    
    def __init__(self, dataset: Dataset, num_samples: int):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for limited dataset of size {self.num_samples}")
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        """Delegate collate_fn to underlying dataset if it exists."""
        if hasattr(self.dataset, 'collate_fn'):
            return self.dataset.collate_fn(batch)
        return batch


class CombinedQuestionDataset(Dataset):
    def __init__(self, datasets: Sequence[BaseQuestionDataset], tokenizer: Any):
        self.datasets = list(datasets)
        self.tokenizer = tokenizer
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int):
        prompt_tokens, _local_idx, seq_len, metadata = self.concat_dataset[idx]
        metadata = {**metadata, "combined_dataset_idx": idx}
        return prompt_tokens, idx, seq_len, metadata

    def collate_fn(self, batch):
        max_seq_len = max(seq_len for _, _, seq_len, _ in batch)
        padded_prompt_tokens = [
            prompt_tokens + [self.tokenizer.pad_id] * (max_seq_len - seq_len)
            for prompt_tokens, _, seq_len, _ in batch
        ]
        return (
            torch.tensor(padded_prompt_tokens, dtype=torch.long),
            torch.tensor([idx for _, idx, _, _ in batch]),
            torch.tensor([seq_len for _, _, seq_len, _ in batch]),
            [metadata for _, _, _, metadata in batch],
        )


def build_combined_question_dataset(
    dataset_names: Sequence[str],
    tokenizer: Any,
    max_token_length: int,
    add_bos_token: bool = False,
    include_choices: bool = True,
    num_samples: int | dict[str, int] | None = None,
) -> CombinedQuestionDataset:
    """
    Build a combined dataset from multiple question datasets.
    
    Args:
        dataset_names: List of dataset names to combine
        tokenizer: Tokenizer instance
        max_token_length: Maximum token length for each sample
        add_bos_token: Whether to add beginning-of-sequence token
        include_choices: Whether to include choices in question prompts when available
        num_samples: Number of samples per dataset. Can be:
            - None: use all samples (default)
            - int: apply same limit to all datasets
            - dict: per-dataset limits, e.g., {"mmlu": 5000, "arc_easy": 100}
    
    Returns:
        CombinedQuestionDataset with the specified datasets
    """
    datasets = []
    for dataset_name in dataset_names:
        dataset_cls = DATASET_REGISTRY[dataset_name]
        dataset = dataset_cls(
            tokenizer=tokenizer,
            max_token_length=max_token_length,
            add_bos_token=add_bos_token,
            include_choices=include_choices,
        )
        
        # Apply num_samples limit if specified
        if num_samples is not None:
            if isinstance(num_samples, dict):
                limit = num_samples.get(dataset_name, len(dataset))
            else:
                limit = num_samples
            dataset = LimitedDataset(dataset, limit)
        
        datasets.append(dataset)
    
    return CombinedQuestionDataset(datasets=datasets, tokenizer=tokenizer)
