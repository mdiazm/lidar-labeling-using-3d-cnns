"""
Loss function used in neural network training
"""

import numpy as np
import torch
import torch.nn.functional as F

def CrossEntropyLoss(pred, ground_truth, smoothing=True):
    """
    Calculate cross entropy loss of amounts of points

    :param pred: predicted labels
    :param ground_truth: real labels of the points
    :param smoothing: (boolean) to smoothe labels or not
    :return: calculated loss
    """

    ground_truth = ground_truth.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss