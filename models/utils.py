"""
    Script to hold utility functions to use in both PointNet & DGCNN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def knn(x, k):
    """
    To get the indexes of the k nearest neighbors of each row in the x matrix.

    Using Euclidean distance.

    :param x: to get the nearest neighbors.
    :param k: number of nearest neighbors to obtain for each point.
    :return: numpy array of dimensions (batch_size, num_points, k)
    """

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # Until here, we have a tensor with the negative square difference of the distances:
    # -[(x1-x2)^2 + (y1-y2)^2]
    pairwise_distance = -1 * pairwise_distance
    pairwise_distance = torch.sqrt(pairwise_distance)

    # Get identifiers of the k nearest neigbors
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Calculate graph from vertexes using features as distance functions.

    :param x: point cloud (or features of the point cloud)
    :param k: number of neighbors to query on each point
    :param idx: identifiers of the points
    :param dim9: boolean. indicates if points are 9-dimensional
    :return: features of size (batch_size, num_points, k)
    """

    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)  # This is not necessary here. Only to ensure that dimensions are the requested ones.

    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, :3], k=k)

    device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx += idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims) -> (batch_size * num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class TransformNet(nn.Module):
    def __init__(self, args):
        super(TransformNet, self).__init__()
        self.args = args
        self.k = 3

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Convolutional layers (cited as shared MLP in paper)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=args.bias),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=args.bias),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=1, bias=args.bias),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Linear layers
        self.linear1 = nn.Linear(1024, 512, bias=args.bias)
        self.linear2 = nn.Linear(512, 256, bias=args.bias)
        self.transform = nn.Linear(256, 3 * 3)  # The output of this model will be the transformation matrix of 3 * 3. Bias is TRUE by default

        init.constant_(self.transform.weight, 0) # Initialize the weight matrix of the last layer to zero. This is feasible because bias is activated in this layer. Otherwise, the model will never converge to the optim
        init.eye_(self.transform.bias.view(3, 3))  # This is not mandatory. View method gets 3*3 dimensional vector as a matrix of dimensions (3, 3)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)  # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn5(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x
