"""
    Attentive Pooling module.

    The purpose of this layer is to infer some output with equal size of the input, and apply a Softmax. This
    will generate a weight vector for each point, with values [0,1] and sum of all of them equals to 1.

    After that, both input and output will be multiplied, applying inferred weights to input features. Finally, all
    of the resulting values are sum and a (batch_size, k_features, num_points) tensor will be returned.

    Pooling will get the most relevant feature in neighbors for n_features.
"""

import torch
from torch import nn
from torch import functional as F

class AttentivePoolingLayer2d(nn.Module):
    def __init__(self, k_features):
        super(AttentivePoolingLayer2d, self).__init__()
        # (batch_size, k_features, num_points, k) -> (batch_size, k_features, num_points)
        self.conv = nn.Conv2d(k_features, k_features, kernel_size=1, bias=False)

        self.softmax = nn.functional.softmax

    def forward(self, x):
        # Forward propagation. Apply attentive pooling
        weights = self.conv(x)

        # Normalize weights to [0, 1] on each feature
        weights = self.softmax(weights, dim=-1)

        # Multiply both input and weights
        new_features = torch.mul(x, weights)

        # Reduce over one dimension using torch.sum
        pooled_features = torch.sum(new_features, dim=-1, keepdim=False)

        return pooled_features

class AttentivePoolingLayer1d(nn.Module):
    def __init__(self, k_features):
        super(AttentivePoolingLayer1d, self).__init__()
        # (batch_size, k_features, num_points) -> (batch_size, k_features, 1)
        self.conv = nn.Conv1d(k_features, k_features, kernel_size=1, bias=False)

        self.softmax = nn.functional.softmax

    def forward(self, x):
        # Forward propagation. Apply attentive pooling
        weights = self.conv(x)

        # Normalize weights to [0, 1] on each feature
        weights = self.softmax(weights, dim=-1)

        # Multiply both input and weights
        new_features = torch.mul(x, weights)

        # Reduce over one dimension using torch.sum
        pooled_features = torch.sum(new_features, dim=-1, keepdim=True)

        return pooled_features
