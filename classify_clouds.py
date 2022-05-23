"""
Cloud Classifier

Classify (or semantically segment) a set of clouds when specifying a dataset. It could be a LiDAR dataset, Semantic 3D
or more (coming).
"""

import argparse
import os
from models.dgcnn import DGCNN
import torch
import utils.neural_handler as neural
from preprocessing.preprocessor import LiDARProcessor, Semantic3DProcessor

BASE_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
DATA_PATH = os.path.join(BASE_PATH, 'data')


def inference(args_p: argparse.Namespace):
    """
    Given a dataset, with some unclassified point clouds, semantically segment all the point clouds contained in
    the dataset.

    :param args_p: input arguments of the classification
    """

    # Create neural handler: class that will hold the load of the models
    model_handler = neural.NeuralHandler(args_p.checkpoint)

    # Load network
    network: torch.nn.Module = model_handler.load_model(args_p.network)

    # Choose compute device
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Device: Using CUDA compute device")
    else:
        device = "cpu"
        print("Device: Using CPU as compute device")

    # Pass model to GPU
    network = network.to(device)

    # Choose dataset processor
    if args_p.dataset == 'Semantic3D':
        processor = Semantic3DProcessor(DATA_PATH)
    elif args_p.dataset == 'LiDAR':
        processor = LiDARProcessor(DATA_PATH)

    # Get available clouds for testing
    clouds = processor.get_test_clouds()

    if len(clouds) == 0:
        print('classify_clouds.py: there is no any unclassified network to classify!')
        exit(-1)

    # Iterate over clouds and process one at once
    for index, cloud in enumerate(clouds):
        # List to store the results of the classification, then postprocess it using some method
        cloud_results = []

        print('classify_clouds.py: classifying cloud with name: ', cloud)

        # Propagate points in blocks
        for point_set, idx in processor.classify_cloud(name=cloud, split_size=args_p.area_size, stride=args_p.stride,
                                                       num_points=args_p.num_points,
                                                       potential_limit=args_p.potential_limit):
            # Pass point set to GPU
            point_set = point_set.to(device)

            # Forward-pass
            prediction = network(point_set)
            prediction = prediction.permute(0, 2, 1).contiguous()

            # Get real classes, not statistical stuff
            predicted_labels = prediction.max(dim=2)[1]
            cloud_results.append((idx, predicted_labels))

        print('classify_clouds.py: cloud classificated with name: ', cloud)
        print('classify_clouds.py: postprocessing cloud: ', cloud)

        # After all, postprocess cloud
        processor.postprocess_cloud(
            name=cloud,
            result=cloud_results,
            voting_scheme='majority'  # TODO review this
        )

        print(
            'classify_clouds.py: postprocessed cloud: {}. {} out of {} processed'.format(cloud, index + 1, len(clouds)))

    print('classify_clouds.py: postprocessed cloud: ', cloud)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference on 3D Point Cloud Dataset")

    # The name of the classification (semantic segmentation)
    parser.add_argument('--name', type=str, default="segmentation")

    # The dataset to use
    parser.add_argument('--dataset', type=str, choices=['Semantic3D', 'LiDAR'], default="Semantic3D")

    # Model that is going to be used for classification
    parser.add_argument('--network', type=str, default=None)

    # Directory of the checkpoints to use
    parser.add_argument('--checkpoint', type=str, default=None)

    # Number of points to choose from each block within each iteration
    parser.add_argument('--num_points', default=4096, type=int)

    # Size (in square meters) of the block
    parser.add_argument('--area_size', type=float, default=1.0)

    # Stride: to use a sliding window approach, and the amount
    parser.add_argument('--stride', type=float, default=1.0)

    # Maximum potential: level at which a point stops to be processed in next iterations
    parser.add_argument('--max_potential', type=float, default=5.0)

    # Step (in potential scale) within each iteration
    parser.add_argument('--step', type=float, default=0.1)

    # Parse args and store them in a dictionary, before starting inference step
    args = parser.parse_args()

    # Classify clouds with the given input args
    inference(args)
