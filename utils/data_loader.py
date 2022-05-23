"""
    Data loaders to create the input pipeline that will serve the data to neural network.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

def feature_filter(points, args):
    """
    Filter input points to use only some features instead of all of them.

    :param points: array of points
    :param args: arguments provided in CLI
    :return: points array but filtered
    """

    idx = list() # Indexes of the features that are going to be selected
    if args.dataset == 'Semantic3D' or args.dataset == 'LiDAR':
        # Points from Semantic3D dataset are like: x y z i r g b xn yn zn
        features = args.features.split(' ')

        if 'x' in features:
            idx.append(0) # Index of 'x' feature
        if 'y' in features:
            idx.append(1) # Index of 'y' feature
        if 'z' in features:
            idx.append(2) # Index of 'z' feature
        if 'i' in features:
            idx.append(3) # Index of 'i' feature
        if 'r' in features:
            idx.append(4) # Index of 'r' feature
        if 'g' in features:
            idx.append(5) # Index of 'g' feature
        if 'b' in features:
            idx.append(6) # Index of 'b' feature
        if 'xn' in features:
            idx.append(7) # Index of 'xn' feature
        if 'yn' in features:
            idx.append(8) # Index of 'yn' feature
        if 'zn' in features:
            idx.append(9) # Index of 'zn' feature

        # Filter input points with selected features only
        filtered_points = points[:, :, np.r_[idx]]

        return filtered_points
    else:
        # TODO Modify to use with other datasets
        pass

def points_filter(points, labels, args):
    """
    Filter number of points. If number of points is 4096, select 2048, for example.

    :param points: numpy array of input points as (batch, points, dims)
    :param labels: labels of the previous points
    :param args: arguments provided by CLI
    :return: filtered array of points containing exactly (batch, points[num_points], dims)
    """

    points_in_batch = points.shape[1]

    # Modify points only if the requested number of points is lower than the real number of points
    if args.num_points < points_in_batch:
        idx = np.arange(0, points_in_batch)

        # Shuffle points to randomly select args.num_points points.
        np.random.shuffle(idx)

        # Select only args.num_points indexes
        idx = idx[:args.num_points]

        # Filter points
        points = points[:, idx, :]

        # Filter labels of the points
        labels = labels[:, idx]

        pass
    elif args.num_points > points_in_batch:
        # TODO augment number of points if requested number is greater than existing one
        pass

    return points, labels

class SemanticSegmentation(Dataset):

    """
    This class wraps the data loading process for training the neural network. As a PyTorch map-style dataset, data
    items are accessed by using a key-value system, where __getitem__() and __len__() methods must be overwritten.
    """

    def __init__(self, points, labels, args, training=True):
        """
        Default constructor of the dataset.
        :param points: numpy array of (num_batches, num_points, dimensions) where num_batches indicates the number of
        data samples in the dataset, and num_points indicates the number of points in each batch. Dimensions is related
        to the dimensions of each point, usually containing (x, y, z, r, g, b, x', y', z').
        :param labels: numpy array of (num_points, 1)
        :param training: if the dataset is being used for training or not
        """

        self.is_training = training

        self.points = points
        self.labels = labels
        self.number_of_batches = self.points.shape[0]
        self.args = args

        # Filter input points and select only key features
        self.points = feature_filter(self.points, self.args)

        # Filter points to use a reduced part (if requested)
        self.points, self.labels = points_filter(self.points, self.labels, self.args)

    def __getitem__(self, item):
        """
        Get batch (bunch) of points by its index and return points separated from their labels.

        :param item: index of the batch of points in self.data numpy array
        :return: (num_points, dimensions) numpy array.
        """
        # Separate points from their associated labels
        points = torch.from_numpy(self.points[item, :]).float()
        # Shuffle points
        indices = list(range(points.shape[0]))
        np.random.shuffle(indices)

        if self.is_training:
            points = points[indices]

        points = points.transpose(1, 0)
        labels = torch.from_numpy(self.labels[item, :]).long()

        if self.is_training:
            labels = labels[indices]
        labels = torch.LongTensor(labels)


        # Return both in separate ways
        return points, labels  # Numpy arrays of (num_points, dimensions - 1) and (num_points,)

    def __len__(self):
        """
        Get length of the dataset (the number of batches containing, not the number of points neither dimensions
        of each).
        :return: (integer) number of batches in the dataset.
        """

        return self.number_of_batches
