"""
SigExt PyTorch Dataset for training with semantic label alignment.

Each token inherits the label of its source sentence, determined by
Sentence-BERT similarity between the sentence and the reference summary.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict


class SigExtDataset(Dataset):
    """Dataset with sentence-to-token label alignment for SigExt training.

    Each token inherits the label of its source sentence. Sentences
    are labeled as salient (1) if their Sentence-BERT similarity to
    the reference summary exceeds the given threshold.

    Args:
        encodings: Tokenizer outputs (input_ids, attention_mask).
        labels: Per-token labels (0 = non-salient, 1 = salient).
    """

    def __init__(self, encodings: Dict, labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)
