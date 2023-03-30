### Michael Engel ### 2022-10-10 ### metrics.py ###
import torch
from torch import Tensor
from sklearn.metrics import cohen_kappa_score


def accuracy(out,targets,mask):
    """
    Computes the accuracy of the model
    
    :param out: Output tensor, representing the predicted class scores for each sample in the batch.
    :type out: Tensor
    :param targets: Target tensor, representing the true class labels for each sample in the batch.
    :type targets: Tensor
    :param mask: Mask tensor, with the same shape as the output tensor, where each element is either 0 or 1.
    :type mask: Tensor
    :return: The accuracy of the model predictions.
    :rtype: Tensor
    """
    pred_classes = torch.argmax(out, axis=1)
    nr_true_pred = torch.eq(pred_classes, targets.squeeze(1))
    return (nr_true_pred.float() * mask.long().squeeze(1)).sum()/(torch.count_nonzero(mask.long()))


def cohen_kappa(out,targets,mask):
    """
    Computes the Cohen's kappa coefficient for the model predictions
    
    :param out: Output tensor, representing the predicted class scores for each sample in the batch.
    :type out: Tensor
    :param targets: Target tensor, representing the true class labels for each sample in the batch.
    :type targets: Tensor
    :param mask: Mask tensor, with the same shape as the output tensor, where each entry is either 0 or 1.
    :type mask: Tensor
    :return: The Cohen's kappa coefficient of the model predictions.
    :rtype: Tensor
    """
    pred_classes = torch.argmax(out, axis=1)
    return cohen_kappa_score(y1=pred_classes.flatten(),y2=targets.flatten(),sample_weight=mask.flatten())


def squared_variance_error(out,targets,mask):
    """
    Computes the squared variance error of the model predictions according to the formula:
    ((var_pred/var_targets) - 1) ** 2
    
    :param out: Output tensor, representing the predicted class scores for each sample in the batch.
    :type out: Tensor
    :param targets: Target tensor, representing the true class labels for each sample in the batch.
    :type targets: Tensor
    :param mask: Mask tensor, with the same shape as the output tensor, where each element is either 0 or 1.
    :type mask: Tensor
    :return: The squared variance error of the model predictions.
    :rtype: Tensor
    """
    # apply mask
    out = out * mask
    targets = targets * mask
    # flatten
    out =  out.reshape(-1)
    targets = targets.reshape(-1)
    # remove
    out = out[out.nonzero()]
    targets = targets[targets.nonzero()]
    # compute var
    var_pred = torch.var(out, unbiased=False)
    var_targets = torch.var(targets, unbiased=False)

    return ((var_pred/var_targets) - 1) ** 2


def masked_mse_loss(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Computes the mean squared error loss with a mask applied to the input and target tensors.
    
    :param input: Input tensor, representing the predicted values for each sample in the batch.
    :type input: Tensor
    :param target: Target tensor, representing the true values for each sample in the batch.
    :type target: Tensor
    :param mask: Mask tensor, with the same shape as the input and target tensors, where each element is either 0 or 1.
    :type mask: Tensor
    :return: The mean squared error loss of the model predictions with the mask applied.
    :rtype: Tensor
    """
    masked_input = input * mask
    masked_target = target * mask
    return torch.nn.functional.mse_loss(masked_input, masked_target) * input.numel() / mask.sum()
