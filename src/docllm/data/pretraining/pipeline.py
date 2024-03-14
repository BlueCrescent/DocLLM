from functools import partial
from typing import List, Tuple

import torch
from torch.utils.data import DataChunk, IterDataPipe
from torch.utils.data.datapipes.iter import FileLister

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
        inputs, bboxes, mask, labels = zip(*items)
        input_batch = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)
        bbox_batch = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True, padding_value=padding_value)
        mask_batch = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=padding_value)
        labels_batch = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=padding_value)
        super().__init__([input_batch, bbox_batch, mask_batch, labels_batch])
