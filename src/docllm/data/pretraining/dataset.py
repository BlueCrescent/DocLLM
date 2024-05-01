import os
from glob import glob
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.masker import DocLLMPretrainingMasker


class DocLLMPretrainDataset(Dataset):
    def __init__(self, config: DocLLMPreTrainDataConfig):
        self._masker = DocLLMPretrainingMasker(config)
        self._files = sorted(glob(os.path.join(config.directory, "**/*.pt"), recursive=True))

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        file_data = torch.load(self._files[idx], map_location="cpu")
        if len(file_data) == 0:
            raise ValueError(f"File is empty.")
        text_tokens, bbox_tokens = tuple(map(list, zip(*file_data)))
        self._check_data(text_tokens, bbox_tokens)
        return self._masker(text_tokens, bbox_tokens)

    def _check_data(self, text_tokens: List[torch.Tensor], bbox_tokens: List[torch.Tensor]) -> None:
        if len(text_tokens) != len(bbox_tokens):
            raise ValueError(
                f"Length of text tokens ({len(text_tokens)}) and bbox tokens ({len(bbox_tokens)}) must match."
            )
        if not all(bbox_tokens[i].size() == (text_tokens[i].size()[0], 4) for i in range(len(text_tokens))):
            raise ValueError("Unexpected bounding box shape.")
