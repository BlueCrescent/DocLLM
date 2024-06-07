from functools import partial
from typing import List, Tuple

import torch
from torch.utils.data import DataChunk, IterDataPipe
from torch.utils.data.datapipes.iter import FileLister

from docllm.data.pretraining.collate import collate
from docllm.data.pretraining.config import DocLLMPreTrainDataConfig


def build_docllm_datapipeline(config: DocLLMPreTrainDataConfig) -> IterDataPipe:
    datapipe = FileLister(config.directory, masks="*.pt", recursive=True)
    if config.shuffle:
        datapipe = datapipe.shuffle(buffer_size=config.shuffle_buffer_size)
    if config.use_sharding_filter:
        datapipe = datapipe.sharding_filter()
    datapipe = datapipe.load_docllm_tensor_data()
    datapipe = datapipe.build_docllm_train_data(config=config)
    if config.use_packing:
        datapipe = datapipe.pack_data(max_sequence_length=config.max_seq_length)
    datapipe = datapipe.batch(
        config.batch_size,
        drop_last=config.drop_last_batch_if_not_full,
        wrapper_class=partial(BatchWithPaddingWrapperClass, padding_value=config.padding_value),
    )
    return datapipe


class BatchWithPaddingWrapperClass(DataChunk):
    def __init__(
        self,
        items: List[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]],
        padding_value: float,
    ):
        super().__init__(collate(items, padding_value))
