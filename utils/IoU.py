"""
Calculate Intersection over Union precision measures.
"""

import torch
import numpy as np

import threading

def IoU_per_class(predictions, labels, num_classes):
    """
    Calculate Intersection over Union for each class given predictions and ground truth labels for these predictions.

    :param predictions: predicted labels using the neural network model, list of tensors. We must concatenate them before doing any calculation.
    :param labels: ground truth for the previous predictions
    :param num_classes: number of classes in the problem
    :return: np array with IoU for each class
    """

    # Concatenate tensors along 0 dim
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    # Turn into one-dimensional arrays by using view()
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    IoU_class = []
    points_class = []
    for i in range(num_classes):
        # Calculate IoU for each class
        # Get predictions whose ground truth is i
        mask_i = labels == i

        # Extract both predictions and ground truth for that class with i index.
        predictions_i = predictions[mask_i]
        labels_i = labels[mask_i]

        # Num points with class i
        points_i = labels_i.shape[0]

        # Calculate IoU for that class
        if points_i > 0:
            iou_class_i = (predictions_i == labels_i).float().sum() / points_i
        else:
            iou_class_i = 0

        # Add both metrics to arrays
        IoU_class.append(iou_class_i)
        points_class.append(points_i)

    return np.array(IoU_class), np.array(points_class)


def calculate_confusion_matrix(predictions, labels, num_classes):
    """
    Calculate confusion matrix given both predictions & labels.

    :param predictions: python list containing predictions
    :param labels: ground truth labels associated to predictions
    :param num_classes: number of classes associated to the prediction problem.
    :return: numpy array containing precision matrix.
    """

    # Concatenate tensors along 0 dim
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    # Turn into one-dimensional arrays by using view()
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    # Create empty zeros matrix to hold predictions: the confusion matrix.
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    # Fill confusion matrix by iterating throughout predictions array
    for i, value in enumerate(predictions):
        # Prediction is correct
        if value == labels[i]:
            confusion_matrix[value][value] += 1
        else:
            confusion_matrix[labels[i]][value] += 1

    return confusion_matrix

class AsyncConfusionMatrix(object):
    def __init__(self, predictions, labels, num_classes, logger, epoch, k, mode):
        self.predictions = predictions
        self.labels = labels
        self.num_classes = num_classes
        self.logger = logger
        self.epoch = epoch
        self.k = k
        self.mode = mode

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        # Concatenate tensors along 0 dim
        self.predictions = torch.cat(self.predictions, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        # Turn into one-dimensional arrays by using view()
        self.predictions = self.predictions.view(-1)
        self.labels = self.labels.view(-1)

        # Create empty zeros matrix to hold predictions: the confusion matrix.
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int)

        # Fill confusion matrix by iterating throughout predictions array
        for i, value in enumerate(self.predictions):
            # Prediction is correct
            if value == self.labels[i]:
                confusion_matrix[value][value] += 1
            else:
                confusion_matrix[self.labels[i]][value] += 1

        self.logger.save_confusion_matrix(confusion_matrix, self.epoch, self.k, self.mode)


