import torch
import torch.nn as nn


class DocLLMCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        losses = super().forward(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses * mask.view(-1).float()
        return losses.sum()
