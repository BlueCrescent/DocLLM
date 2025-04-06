import logging
import multiprocessing
import os
import random
from typing import Iterable, List, Protocol, Tuple

import torch


class SampleBuilder(Protocol):
    def __call__(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        pass


class SampleBuilderFactory(Protocol):
    def __call__(self) -> SampleBuilder:
        pass


def precompute_epoch_files(
    directory: str,
    output_directory: str,
    max_seq_length: int,
    num_processes: int,
    sample_builder_factory: SampleBuilderFactory,
    start_idx: int = 0,
    end_idx: int = -1,
) -> None:
    files = sorted(list_pt_files(directory))[start_idx:end_idx]
    random.shuffle(files)
    queue = multiprocessing.Queue()
    for file in files:
        queue.put(file)

    workers = [
        multiprocessing.Process(
            target=precompute_worker,
            args=(queue, output_directory, max_seq_length, f"subset_{i}_file", sample_builder_factory),
        )
        for i in range(num_processes)
    ]
    try:
        for worker in workers:
            worker.start()
        for _ in range(num_processes):
            queue.put(None)
    except Exception as e:
        print(f"Error while precomputing files: {e}")
        for worker in workers:
            worker.terminate()
        raise e
    finally:
        for worker in workers:
            worker.join()


def precompute_worker(
    queue: multiprocessing.Queue,
    output_directory: str,
    max_seq_length: int,
    prefix: str,
    sample_builder_factory: SampleBuilderFactory,
) -> None:
    precomputer = SubsetPrecomputer(output_directory, max_seq_length, prefix, sample_builder_factory)
    precomputer.process(IterableQueue(queue))


class SubsetPrecomputer:
    def __init__(
        self, output_directory: str, max_seq_length: int, prefix: str, sample_builder_factory: SampleBuilderFactory
    ):
        self._output_directory = output_directory
        self._max_sequence_length = max_seq_length
        self._prefix = prefix
        self._sample_builder: SampleBuilder = sample_builder_factory()
        self._elements = []
        self._total_size = 0

    def process(self, files: Iterable[str]) -> None:
        self._save_packed_data(self._build_packed_data(self._build_training_inputs(self._load_pickle_file(files))))

    def _load_pickle_file(self, file_paths: Iterable[str]) -> Iterable[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        for path in file_paths:
            file_data = torch.load(path, map_location="cpu", weights_only=True)
            if len(file_data) == 0:
                print(f"File '{path}' is empty.")
                continue
            text_tokens, bbox_tokens = tuple(map(list, zip(*file_data)))
            if self._check_data(text_tokens, bbox_tokens, path):
                yield text_tokens, bbox_tokens

    def _check_data(self, text_tokens: List[torch.Tensor], bbox_tokens: List[torch.Tensor], path: str) -> bool:
        if len(text_tokens) != len(bbox_tokens):
            logging.warning(
                f"Length of text tokens ({len(text_tokens)}) and bbox tokens "
                f"({len(bbox_tokens)}) must match. (In file {path})"
            )
            return False
        if not all(bbox_tokens[i].size() == (text_tokens[i].size()[0], 4) for i in range(len(text_tokens))):
            logging.warning("Unexpected bounding box shape. (In file {path})")
            return False
        return True

    def _build_training_inputs(
        self, input_and_bbox_tensors: Iterable[Tuple[List[torch.Tensor], List[torch.Tensor]]]
    ) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]:
        for input_tensors, bbox_tensors in input_and_bbox_tensors:
            try:
                yield self._sample_builder(input_tensors, bbox_tensors)
            except ValueError as e:
                logging.warning(e)

    def _build_packed_data(
        self, file_data: Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]
    ) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]:
        for item in file_data:
            item_size = item[0].size(0)
            if self._total_size + item_size > self._max_sequence_length:
                yield self._build_packed_entry(self._elements)
                self._elements = []
                self._total_size = 0
            self._elements.append(item)
            self._total_size += item_size

        if self._elements:
            yield self._build_packed_entry(self._elements)

    def _build_packed_entry(
        self, elemets: List[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]]
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]:
        inputs, bboxes, mask, labels = zip(*elemets)
        sizes = torch.LongTensor([i.size(0) for i in inputs])
        return torch.cat(inputs), torch.cat(bboxes), torch.cat(mask), torch.cat(labels), sizes

    def _save_packed_data(
        self,
        packed_data: Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.LongTensor]],
    ) -> None:
        for i, data in enumerate(packed_data):
            new_fn = os.path.join(self._output_directory, self._prefix + f"_{i}.pt")
            torch.save(data, new_fn)


class IterableQueue:
    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue

    def __iter__(self):
        while True:
            task = self.queue.get()
            if task is None:
                break
            yield task


def list_pt_files(directory: str) -> List[str]:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
