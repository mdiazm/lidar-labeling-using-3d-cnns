"""
Default script to train a Neural Network (and retrain it for transfer learning). Every setting of the training process
is passed using argument parser, so this script will be executed from the command line.
"""

# System imports
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime as dt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score as measure_accuracy

# Custom imports
from models.dgcnn import DGCNN, weights_init
from utils.loss import CrossEntropyLoss
from preprocessing.preprocessor import LiDARProcessor, Semantic3DProcessor
from utils.logger import TensorboardLogger
from utils.data_loader import SemanticSegmentation
from utils.IoU import IoU_per_class, calculate_confusion_matrix, AsyncConfusionMatrix
from utils.neural_handler import NeuralHandler
from utils.measurers import TimeMeasurer

from utils.viewers import visualize_cloud_splitted, visualize_cloud

BASE_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
DATA_PATH = os.path.join(BASE_PATH, 'data')

def init(args):
    """
    Create directories to save experiment results.

    :param args: dictionary containing the whole parameters of the experiment.
    :return tensorboard logger reference.
    """

    # Experiments directory
    experiments_dir = os.path.join(BASE_PATH, 'experiments')
    this_experiment = os.path.join(experiments_dir, args.exp_name)
    checkpoints = os.path.join(this_experiment, args.output_models)

    # Check if 'experiments' directory exists on base path
    if not os.path.isdir(experiments_dir):
        print("Created 'experiments' directory")
        os.mkdir(experiments_dir)

    # Check if exists a directory for this experiment
    if not os.path.isdir(this_experiment):
        print("Created 'experiments/{}' directory".format(args.exp_name))
        os.mkdir(this_experiment)
    else:
        print("Existing directory for experiment '{}', aborting creation...".format(args.exp_name))

    # Directory for checkpoints (trained models)
    if not os.path.isdir(checkpoints):
        print("Created 'experiments/{}/{}' directory".format(args.exp_name, args.output_models))
        os.mkdir(checkpoints)

    # Create tensorboard logger
    logger = TensorboardLogger(this_experiment, args)

    return logger

def train(args, logger=None):
    """
    Network training function.

    :param args: dictionary containing the parameters specified using command line.
    """

    # Start message
    d_time = str(dt.today().strftime('%Y-%m-%d-%H:%M:%S'))
    print("Starting experiment '{}' on time {}".format(args.exp_name, d_time))

    experiments_dir = os.path.join(BASE_PATH, 'experiments')
    this_experiment = os.path.join(experiments_dir, args.exp_name)
    checkpoints = os.path.join(this_experiment, args.output_models)

    # Create model handler to store trained models
    neural_handler = NeuralHandler(checkpoints)

    # Choose compute device
    if torch.cuda.is_available():
        device = "cuda:0"
        print("Device: Using CUDA compute device")
    else:
        device = "cpu"
        print("Device: Using CPU as compute device")

    # Loss function (modified cross entropy)
    criterion = CrossEntropyLoss
    criterion2 = nn.CrossEntropyLoss()
    print("Loss function: Using Cross Entropy Loss for this experiment")

    # Create processor (and use load_data() method to employ kfold cross validation by iterating)
    if args.dataset == 'Semantic3D':
        print("Dataset: Training with Semantic3D dataset")
        processor = Semantic3DProcessor(DATA_PATH, training_items=args.input_data)
    elif args.dataset == 'LiDAR':
        print("Dataset: Training with a LiDAR dataset")
        processor = LiDARProcessor(DATA_PATH, training_items=args.input_data)

    # Get number of classes of the problem to adjust neural network
    args.num_classes = len(processor.labels)

    # Create neural network model
    network: DGCNN = None
    if args.retrain:
        network = neural_handler.load_model_transfer_learning(args.checkpoint, args.num_classes)
        network = network.to(device)
    else:
        network = DGCNN(args).to(device)
        network = nn.DataParallel(network)

    # Add network architecture to tensorboard
    # logger.add_graph(network) #TODO review this

    # If several GPUs are available, distribute model on them
    print("Devices: {} GPUs are going to be used".format(torch.cuda.device_count()))

    # Create optimizer
    if args.optimizer == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        print("Optimizer: using Adam optimizer")
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)
        print("Optimizer: using Stochastic Gradient Descent optimizer")

    # Variables to store information about the metrics on each partition
    train_kfold_loss = []
    train_kfold_accuracy = []
    test_kfold_loss = []
    test_kfold_accuracy = []

    # Iterate through the dataset (for each one of the partitions)
    for i, ((train_data, train_labels), (test_data, test_labels)) in enumerate(processor.load_data()):
        # K Fold Cross Validation
        print("K-Fold Cross Validation: partition {}".format(i + 1))

        # Do training for each (train, test) partition as kFold. So, run all the epochs starting from zero
        # Set same seed. This guarantees that the model will start always in the same solution-space, so good results due to randomness of partitioning are not suitable here.
        torch.manual_seed(1)

        # Re-initialize model weights by using next function
        network.apply(weights_init)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)  # 10 means that each 10 epochs, lr is reduced to its 95%

        # Create both training & test datasets
        train_dset = SemanticSegmentation(train_data, train_labels, args) # TODO take care about this!!!!!!!
        test_dset = SemanticSegmentation(test_data, test_labels, args, training=False)

        # Data loaders for training and test datasets
        train_loader = DataLoader(train_dset, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dset, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

        for epoch in range(args.epochs):

            # Variables to store metrics
            train_epoch_loss = 0.0
            train_epoch_accuracy = 0.0
            test_epoch_loss = 0.0
            test_epoch_accuracy = 0.0
            num_batches = 0
            num_points = 1
            prediction_tensors = []
            ground_truth_tensors = []

            # Training loop
            print("Performing training on epoch {}/{} and partition {}".format(epoch + 1, args.epochs, i + 1))
            network.train()
            for points, labels in train_loader:
                # Copy data to compute device
                points, labels = points.to(device), labels.to(device)

                # Batch size
                batch_size = points.size()[0]

                # Number of points in each batch
                num_points = points.size()[2] # (batch, dim, points)

                # Set gradients to zero
                optimizer.zero_grad()

                # Get prediction from network
                prediction = network(points)
                prediction = prediction.permute(0, 2, 1).contiguous()

                # loss = criterion2(prediction.view(-1, args.num_classes), labels.view(-1))

                # Evaluate prediction --> calculate loss
                loss = criterion(prediction.view(-1, args.num_classes), labels.view(-1, 1).squeeze())
                loss.backward()

                # Update weights
                optimizer.step()

                # Add loss to general loss
                train_epoch_loss += loss.item() * batch_size

                # Calculate accuracy
                predicted_labels = prediction.max(dim=2)[1]
                # accuracy = measure_accuracy(labels.view(-1).cpu().numpy(), predicted_labels.view(-1).cpu().numpy())
                accuracy = (predicted_labels == labels).float().sum()

                # TODO only for debugging
                # print('Ground truth: {}'.format(np.unique(labels.view(-1).cpu().numpy())))
                # print('Predicted ones: {}'.format(np.unique(predicted_labels.view(-1).cpu().numpy())))

                train_epoch_accuracy += accuracy

                # Increase number of mini-batches
                num_batches += batch_size

                # Calculate IoU per each class
                prediction_tensors.append(predicted_labels.to('cpu'))
                ground_truth_tensors.append(labels.to('cpu'))

            # Calculate real loss & accuracy (average loss & accuracy)
            train_epoch_loss /= num_batches
            train_epoch_accuracy /= (num_batches * num_points) # divide accumulated accuracy / total number of points
            train_iou_per_class, train_num_points_class = IoU_per_class(prediction_tensors, ground_truth_tensors, args.num_classes)

            # Calculate confussion matrix (synchronous)
            # confusion_matrix = calculate_confusion_matrix(prediction_tensors, ground_truth_tensors, args.num_classes)
            # logger.save_confusion_matrix(confusion_matrix, epoch + 1, i + 1, 'training')

            # Calculate confussion matrix (asynchronous)
            async_confusion = AsyncConfusionMatrix(prediction_tensors, ground_truth_tensors, args.num_classes, logger, epoch + 1, i + 1, 'training')

            # Validation loop (test)
            print("Performing evaluation on epoch {}/{} and partition {}".format(epoch + 1, args.epochs, i + 1))
            # We must restart num_batches for re-initializing
            num_batches = 0
            prediction_tensors = []
            ground_truth_tensors = []
            network.eval()
            for points, labels in test_loader:
                # Copy data to compute device
                points, labels = points.to(device), labels.to(device)

                # Batch size
                batch_size = points.size()[0]

                # Number of points
                num_points = points.size()[2]

                # Get prediction from network
                prediction = network(points)
                prediction = prediction.permute(0, 2, 1).contiguous()

                # Evaluate prediction --> calculate loss
                loss = criterion(prediction.view(-1, args.num_classes), labels.view(-1, 1).squeeze())

                # Add loss to general loss
                test_epoch_loss += loss.item() * batch_size

                # Calculate accuracy
                predicted_labels = prediction.max(dim=2)[1]
                # accuracy = measure_accuracy(labels.view(-1).cpu().numpy(), predicted_labels.view(-1).cpu().numpy())
                accuracy = (predicted_labels == labels).float().sum() # Accuracy is measured as the number of predictions that are equal to the ground truth
                test_epoch_accuracy += accuracy

                # Increase number of mini-batches
                num_batches += batch_size

                # Calculate IoU per each class
                prediction_tensors.append(predicted_labels.to('cpu'))
                ground_truth_tensors.append(labels.to('cpu'))

            # Calculate real loss & accuracy (average loss & accuracy)
            test_epoch_loss /= num_batches
            test_epoch_accuracy /= (num_batches * num_points) # divide accumulated accuracy / total number of points
            test_iou_per_class, test_num_points_class = IoU_per_class(prediction_tensors, ground_truth_tensors, args.num_classes)

            # Calculate confussion matrix
            # confusion_matrix = calculate_confusion_matrix(prediction_tensors, ground_truth_tensors, args.num_classes)
            # logger.save_confusion_matrix(confusion_matrix, epoch + 1, i + 1, 'validation')

            # Calculate confussion matrix (asynchronous)
            async_confusion = AsyncConfusionMatrix(prediction_tensors, ground_truth_tensors, args.num_classes, logger, epoch + 1, i + 1, 'validation')

            # Update learning rate
            scheduler.step()

            # Print statistics
            log = "Partition {} | epoch {}/{} | train loss {} | train accuracy {} | test loss {} test accuracy {}"
            log = log.format(i + 1, epoch + 1, args.epochs, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy)
            print(log)

            # Register metrics on tensorboard
            logger.add_scalar('loss/train', train_epoch_loss, epoch + 1, kfold=True, k=i+1)
            logger.add_scalar('loss/test', test_epoch_loss, epoch + 1, kfold=True, k=i+1)
            logger.add_scalar('accuracy/train', train_epoch_accuracy, epoch + 1, kfold=True, k=i+1)
            logger.add_scalar('accuracy/test', test_epoch_accuracy, epoch + 1, kfold=True, k=i+1)

            # Add pairs of scalars, so it is possible to compare them
            logger.add_scalars('loss', {
                'train': train_epoch_loss,
                'test': test_epoch_loss
            }, epoch + 1, kfold=True, k=i+1)

            logger.add_scalars('accuracy', {
                'train': train_epoch_accuracy,
                'test': test_epoch_accuracy
            }, epoch + 1, kfold=True, k=i+1)

            logger.add_scalars('train', {
                'loss': train_epoch_loss,
                'accuracy': train_epoch_accuracy
            }, epoch + 1, kfold=True, k=i+1)

            logger.add_scalars('test', {
                'loss': test_epoch_loss,
                'accuracy': test_epoch_accuracy
            }, epoch + 1, kfold=True, k=i+1)

            # Add IoU for each class and mean IoU (accuracy metric in this work)
            # Training IoU
            iou_train = {}
            for i_class, iou_class in enumerate(train_iou_per_class):
                iou_train['class_' + str(i_class)] = iou_class

            iou_train['mean_IoU'] = train_epoch_accuracy
            logger.add_scalars('iou/train', iou_train, epoch + 1, kfold=True, k=i+1)

            # Test IoU
            iou_test = {}
            for i_class, iou_class in enumerate(test_iou_per_class):
                iou_test['class_' + str(i_class)] = iou_class

            iou_test['mean_IoU'] = test_epoch_accuracy
            logger.add_scalars('iou/test', iou_test, epoch + 1, kfold=True, k=i+1)

            # Store trained model per epoch and K fold
            model_name = "{expname}-network-epoch-{epoch}-k-{k}".format(
                expname=args.exp_name,
                epoch=epoch + 1,
                k=i+1
            )
            neural_handler.store_model(network=network, name=model_name, args=args)

        # When training finishes, save metrics for each partition in the last epoch: .item() because are tensors on gpu
        train_kfold_accuracy += [train_epoch_accuracy.item()]
        train_kfold_loss += [train_epoch_loss]
        test_kfold_accuracy += [test_epoch_accuracy.item()]
        test_kfold_loss += [test_epoch_loss]

        # Points on each class
        logger.add_text('points_per_class/train', text=str(train_num_points_class), kfold=True, k=i+1)
        logger.add_text('points_per_class/test', text=str(test_num_points_class), kfold=True, k=i+1)

    # Calculate average metrics of the whole set of partitions
    train_avg_accuracy = np.mean(np.array(train_kfold_accuracy))
    train_avg_loss = np.mean(np.array(train_kfold_loss))
    test_avg_accuracy = np.mean(np.array(test_kfold_accuracy))
    test_avg_loss = np.mean(np.array(test_kfold_loss))

    # Save metrics to tensorboard
    logger.add_scalar('avg/loss/train', train_avg_loss, args.epochs)
    logger.add_scalar('avg/accuracy/train', train_avg_accuracy, args.epochs)
    logger.add_scalar('avg/loss/test', test_avg_loss, args.epochs)
    logger.add_scalar('avg/accuracy/test', test_avg_accuracy, args.epochs)

    # Print metrics
    log = "Finished experiment '{}' | train loss {} | train accuracy {} | test loss {} | test accuracy {}"
    log = log.format(args.exp_name, train_avg_loss, train_avg_accuracy, test_avg_loss, test_avg_accuracy)
    print(log)

if __name__ == '__main__':
    # Main function of the training process
    parser = argparse.ArgumentParser(description="Point Cloud Semantic Segmentation using DGCNN")

    # The name of the experiment
    parser.add_argument('--exp_name', type=str, default="transfer", help="Name of the experiment (and the trained model)")

    # The dataset to use
    parser.add_argument('--dataset', type=str, required=False, help="Dataset to use", choices=["Semantic3D", "LiDAR"], default="LiDAR")

    # Input data to use: default train_items
    parser.add_argument('--input_data', type=str, required=False, help="Input data to use", default='train_items')

    # Learning rate of the neural network training
    parser.add_argument('--learning_rate', type=float, help="The (initial) learning rate to use (default 0.001)", default=0.001)

    # Learning rate decreases
    parser.add_argument('--scheduler', type=bool, help="Wether or not to use scheduler in leaning rate", default=True)

    # Epochs of training
    parser.add_argument('--epochs', type=int, help="The number of epochs in the experiment (default 100)", default=100)

    # Nearest neighbors to consider when generating the graph (feature graph)
    parser.add_argument('--nearest_neighbors', type=int, help="Number of nearest neighbors to consider during network training (default 15)", default=15, required=False)

    # Momentum
    parser.add_argument('--momentum', type=float, help="Momentum associated to SGD", default=0.9)

    # The directory (name of it) to save trained models (weight & biases)
    parser.add_argument('--output_models', type=str, help="Where to save the trained models (name of directory)", default="checkpoints")

    # The optimizer to use during training
    parser.add_argument('--optimizer', type=str, choices=["Adam", "SGD"], default="SGD", help="Optimizer to use during training")

    # Use bias in neural network layers
    parser.add_argument('--bias', type=bool, help="To use bias on neural network layers or not", default=False)

    # Points are 9-dimensional
    parser.add_argument('--dim9', type=bool, help="Points are 9 dimensional in NN pipeline", default=True)

    # Dimensions of the embeddings
    parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings')

    # To use dropout or not
    parser.add_argument('--dropout', type=float, help="Use dropout layers in NN (default 0)", default=0)

    # Number of workers (threads) to use during train/test in DataLoaders
    parser.add_argument('--workers', type=int, help="Number of workers to use in DataLoaders", default=8)

    # Batch size of the experiment
    parser.add_argument('--batch_size', type=int, help="The batch size of the experiment (default 2)", default=4)

    # Features to use
    parser.add_argument('--features', type=str, help="The features to use: x y z r g b i xn yn zn", default="x y z i r g b")

    # Number of features that are being used
    parser.add_argument('--num_features', type=int, help="The number of features that are being considered for constructing the model", default=7)

    # Number of points
    parser.add_argument('--num_points', type=int, help="Number of points used in the experiment", default=2048)

    # Type of pooling used
    parser.add_argument('--pooling', type=str, choices=['max', 'attentive'], help='Pooling layers that are going to be used' ,default='max')

    # Transfer learning
    parser.add_argument('--retrain', type=bool, help="To retrain a model from a checkpoint", default=False)

    # Checkpoint to use in transfer learning
    parser.add_argument('--checkpoint', type=str, help="Pretrained model to use in transfer learning (name)", default="geometry_i_rgb_base")

    # Parse args and store them in a dictionary "args"
    args = parser.parse_args()

    # Create main directories & get reference to TensorboardLogger
    logger = init(args)

    # Call to train function
    train(args, logger)
