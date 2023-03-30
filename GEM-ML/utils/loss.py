import torch
from torch import Tensor

class MSELossMasked(torch.nn.MSELoss):
    """
    This class provides the mean squared error (MSE) loss function implementation in PyTorch,
    which is extended to handle masked input tensors.
    """
    
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELossMasked, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Computes the mean squared error between the masked input and target tensors.

        :param input: Input tensor.
        :type input: Tensor
        :param target: Target tensor.
        :type target: Tensor
        :param mask: Mask tensor, with the same shape as the input tensor, where each element is either 0 or 1.
        :type mask: Tensor
        :return: Scalar loss tensor.
        :rtype: Tensor
        """
        masked_input = input * mask
        masked_target = target * mask
        return super().forward(masked_input, masked_target) * input.numel() / mask.sum()
