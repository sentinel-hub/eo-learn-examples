### Michael Engel ### 2022-10-10 ### metrics.py ###
import torch
from sklearn.metrics import cohen_kappa_score

def accuracy(out,targets,mask):
    pred_classes = torch.argmax(out, axis=1)
    nr_true_pred = torch.eq(pred_classes, targets.squeeze(1))
    return (nr_true_pred.float() * mask.long().squeeze(1)).sum()/(torch.count_nonzero(mask.long()))

def cohen_kappa(out,targets,mask):
    pred_classes = torch.argmax(out, axis=1)
    return cohen_kappa_score(y1=pred_classes.flatten(),y2=targets.flatten(),sample_weight=mask.flatten())