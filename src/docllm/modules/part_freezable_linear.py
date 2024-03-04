import torch
import torch.nn as nn


class PartFreezableLinear(nn.Linear):
    """
    A linear layer with the ability to freeze the original outputs and add additional outputs.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_additional_outputs (int, optional): Number of additional outputs to be added. Default is 0.

    Methods:
        set_freeze_original_outputs(freeze: bool): Sets whether to freeze the original outputs.
        forward(input: torch.Tensor) -> torch.Tensor: Performs forward pass of the linear layer.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_additional_outputs: int = 0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self._num_additional_outputs = num_additional_outputs
        if self._num_additional_outputs > 0:
            self._additional_outputs = nn.Linear(
                in_features=in_features,
                out_features=self._num_additional_outputs,
                bias=bias,
            )

    def set_freeze_original_outputs(self, freeze: bool):
        """
        Sets whether to freeze the original outputs.

        Args:
            freeze (bool): If True, the original outputs will be frozen. If False, the original outputs will be trainable.

        """
        self.requires_grad_(not freeze)
        if self._num_additional_outputs > 0:
            self._additional_outputs.requires_grad_(True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the linear layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        output = super().forward(input)
        if self._num_additional_outputs > 0:
            additional_output = self._additional_outputs(input)
            output = torch.cat([output, additional_output], dim=-1)
        return output
