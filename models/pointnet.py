"""
    This script contains code for PointNet neural network
"""

import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    """
    PointNet Classification & Segmentation model definition
    """
    def __init__(self, args, output_channels=40, bias=False):
        super(PointNet, self).__init__()
        self.args = args

        # One dimensional convolutional layers. In paper: MLP shared
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=bias)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=bias)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)

        # Linear layers (fully-connected ones)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=bias)
        self.linear2 = nn.Linear(512, output_channels)

        # Dropout layers
        self.dp1 = nn.Dropout()  # By default, p = 0.5

        def forward(self, x):

            # 1st layer
            x = F.relu(self.bn1(self.conv1(x)))

            # 2nd layer
            x = F.relu(self.bn2(self.conv2(x)))

            # 3rd layer
            x = F.relu(self.bn3(self.conv3(x)))

            # 4th layer
            x = F.relu(self.bn4(self.conv4(x)))

            # 5th layer
            x = F.relu(self.bn5(self.conv5(x)))

            # Symmetric function: max pooling
            x = F.adaptive_max_pool1d(x, 1).squeeze()

            # Linear layer
            x = F.relu(self.bn6(self.linear1(x)))

            # Dropout
            x = self.dp1(x)

            # Last linear layer
            x = self.linear2(x)

            return x