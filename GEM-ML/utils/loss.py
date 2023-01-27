import torch
from torch import Tensor


class MSELossMasked(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELossMasked, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        masked_input = input * mask
        masked_target = target * mask
        return super().forward(masked_input, masked_target) * input.numel() / mask.sum()
