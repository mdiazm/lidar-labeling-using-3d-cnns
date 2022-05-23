"""
    Script to prepare data: from point cloud to blocks of semantized points.
"""

import os
from preprocessing.preprocessor import Semantic3DProcessor
from preprocessing.preprocessor import LiDARProcessor
import argparse
import sys

BASE_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
DATA_PATH = os.path.join(BASE_PATH, 'data')

if __name__ == '__main__':
    # Main function of the training process
    parser = argparse.ArgumentParser(description="Point Cloud Semantic Segmentation using DGCNN: data preprocessing")

    # To download data or not
    parser.add_argument('--download', type=bool, default=False, help="To download dataset from the internet")

    # The dataset to use
    parser.add_argument('--dataset', type=str, default="LiDAR", help="The dataset to prepare data from")

    # To redirect output data (prints) to a external file instead of console
    parser.add_argument('--redirect_log', type=str, default=None,
                        help="The path to the file where the output log will be written. If None, console is used")

    # To subsample dataset or not
    parser.add_argument('--subsample', type=bool, default=False, help="To subsample dataset or not during execution")

    # Choice of k-fold cross validation
    parser.add_argument('--kfold', type=int, default=2, help="K used to make partitions for K Fold Cross Validation")

    # Choice of split size (during point cloud block generation)
    parser.add_argument('--split_size', type=float, default=4.0, help="Size of blocks which contain points (meters)")

    # Number of points selected in each block
    parser.add_argument('--num_points', type=int, default=2048, help="Number of points selected in each block")

    # Min number of points to consider a block. If real number of points is lower, the points will be duplicated
    parser.add_argument('--min_points', type=int, default=512, help="Minimum amount of points to consider a block")

    # Number of points selected in each block
    parser.add_argument('--resample', type=bool, default=False, help="To sample point cloud again")

    # Stride used to move sliding windows across the plane defined by the point cloud when generating blocks
    parser.add_argument('--stride', type=float, default=4.0, help="Stride of the sliding window approach")

    # Output train items directory
    parser.add_argument('--output_dir', type=str, default='train_items', help="Output directory for training items")

    # Parse arguments
    args = parser.parse_args()

    # Redirect standard output to file if needed
    if args.redirect_log is not None:
        sys.stdout = open(args.redirect_log, 'w')

    # Choice of dataset
    if args.dataset == 'Semantic3D':
        processor = Semantic3DProcessor(DATA_PATH, download=args.download, training_items=args.output_dir)
    elif args.dataset == 'LiDAR':
        processor = LiDARProcessor(DATA_PATH, training_items=args.output_dir)
    else:
        processor = None

    # Subsample dataset if needed
    if args.resample:
        if args.subsample:
            processor.subsample_dataset(method='random')
        else:
            processor.subsample_dataset(method=None)

    # Make partitions (generate batches)
    processor.prepare_training_dataset(
        k=args.kfold,
        split_size=args.split_size,
        num_points=args.num_points,
        stride=args.stride,
        min_points=args.min_points
    )
