"""
    Dynamic Graph CNN code (for semantic segmentation of 3D scenes only!)
"""

from models.utils import get_graph_feature
from models.utils import TransformNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attentive_pooling import AttentivePoolingLayer2d, AttentivePoolingLayer1d

def weights_init(m):
    """
    Callback. Reinitialize weights of a given neural net.

    :param m: each one of the neural network layers. This is internally done by definition.
    """
    # Two-dimensional convolutional layers
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)

    # One-dimensional convolutional layers
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)

class DGCNN(nn.Module):
    """
    Dynamic Graph CNN model
    """

    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.nearest_neighbors
        self.dim9 = args.dim9
        self.num_classes = args.num_classes

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Spatial Transform Network
        self.stn = TransformNet(args)

        # Convolutional (shared MLP) layers

        """
            BACKBONE: feature extraction
        """
        # 1st EdgeConv Block
        # 1st convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(args.num_features * 2, 64, kernel_size=1, bias=args.bias),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 2nd convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=args.bias),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.edge_conv1 = nn.Sequential(
            self.conv1,
            self.conv2
        )

        self.attentive1 = AttentivePoolingLayer2d(64)

        # 2nd EdgeConv Block
        # 3rd convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=args.bias),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 4th convolutional layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=args.bias),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.edge_conv2 = nn.Sequential(
            self.conv3,
            self.conv4
        )

        self.attentive2 = AttentivePoolingLayer2d(64)

        # 3rd (and last) EdgeConv Block
        # 5th convolutional layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=args.bias),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.edge_conv3 = nn.Sequential(
            self.conv5
        )

        self.attentive3 = AttentivePoolingLayer2d(64)

        # 6th convolutional layer: gets word descriptor of each cloud fragment
        self.conv6 = nn.Sequential(
            nn.Conv1d(64 * 3, args.emb_dims, kernel_size=1, bias=args.bias),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.attentive4 = AttentivePoolingLayer1d(args.emb_dims)

        """
            HEAD: classification (semantic segmentation) of points
        """
        # 7th convolutional layer
        self.conv7 = nn.Sequential(
            nn.Conv1d(1216, 512, kernel_size=1, bias=args.bias),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 8th convolutional layer
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=args.bias),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 9th convolutional layer
        self.conv9 = nn.Conv1d(256, args.num_classes, kernel_size=1, bias=args.bias)

        # Dropout layer
        self.dp1 = nn.Dropout(p=args.dropout)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # Spatial Transform Network
        # x = self.stn(x)

        # 1st EdgeConv
        x = get_graph_feature(x, k=self.k, dim9=self.dim9)  # (batch_size, dims, num_points) -> (batch_size, dims*2, num_points, k) -> x2 because both features of points and difference of neighbours with points are concat.
        x = self.conv1(x)  # (batch_size, dims*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)

        if self.args.pooling == 'max':
            x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points) | dim = -1 means same that dim = 3 in the case of the tensor has 4 dimensions (like array slicing for selecting the last element). Keepdim automatically squeezes last dimension when reduced. That is, instead of having [batch_size, dims, points, 1], delete that one last dimension.
        else:
            x1 = self.attentive1(x)
        ############## [0] select only the values and not the indices

        # 2nd EdgeConv
        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)

        if self.args.pooling == 'max':
            x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        else:
            x2 = self.attentive2(x)
        ##############

        # 3rd EdgeConv
        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)

        if self.args.pooling == 'max':
            x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        else:
            x3 = self.attentive3(x)
        ##############

        # 1st concatenation
        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)
        ##############

        # 1st Block: Extract word descriptor of the cloud fragment
        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)

        if self.args.pooling == 'max':
            x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        else:
            x = self.attentive4(x)
        ##############

        # 2nd Block
        x = x.repeat(1, 1, num_points)  # (batch_size, emb_dims[1024], num_points) # word vector is replicated for each point in the cloud segment. So, each point will contain not only its local information but also global information.

        # 2nd concatenation
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        # Last block
        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)  # Not modifies tensor dimensions
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, num_classes, num_points)

        return x

    def get_config(self):
        """
        Get a dict of the config used in the model: kNearest, dim9, etc.
        :return: dict (pair, value) with the config of the network
        """

        return {
            'args': self.args,
            'k': self.k,
            'dim9': self.dim9,
            'classes': self.num_classes
        }

    def change_number_output_classes(self, number_classes):
        self.conv9 = nn.Conv1d(256, number_classes, kernel_size=1, bias=self.args.bias)