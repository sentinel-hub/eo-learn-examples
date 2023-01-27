### Michael Engel ### 2022-10-10 ### metrics.py ###
import torch
from torch import Tensor
from sklearn.metrics import cohen_kappa_score


def accuracy(out,targets,mask):
    pred_classes = torch.argmax(out, axis=1)
    nr_true_pred = torch.eq(pred_classes, targets.squeeze(1))
    return (nr_true_pred.float() * mask.long().squeeze(1)).sum()/(torch.count_nonzero(mask.long()))


def cohen_kappa(out,targets,mask):
    pred_classes = torch.argmax(out, axis=1)
    return cohen_kappa_score(y1=pred_classes.flatten(),y2=targets.flatten(),sample_weight=mask.flatten())


def squared_variance_error(out,targets,mask):
    """
    Calculates the squared variance error:
    ((var_pred/var_targets) - 1) ** 2
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
    masked_input = input * mask
    masked_target = target * mask
    return torch.nn.functional.mse_loss(masked_input, masked_target) * input.numel() / mask.sum()
