"""
Script containing a tensorboard-based logger
"""

import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TensorboardLogger:
    """
    Tensorboard-based logger to visualize results of the experiments. Each experiment will have its own runs/
    directory, where tensorboard will be stored.
    """

    def __init__(self, exp_path, args=None):
        """
        Constructor of the class

        :param exp_path: absolute path of the experiment directory.
        :param args: arguments/parameters of the experiment. #TODO later
        """

        # runs/ directory path
        self.runpath = os.path.join(exp_path, 'runs')
        self.confusion_matrices = os.path.join(exp_path, 'matrices')
        self.args = args

        # Extract experiment name
        self.exp_name = exp_path.split('/')[-1]

        # Check if runs/ directory exists
        if os.path.isdir(self.runpath):
            print("Directory '{}' exists. Aborting creation...".format(self.runpath))

        if os.path.isdir(self.confusion_matrices):
            print("Directory '{}' exists. Aborting creation...".format(self.confusion_matrices))
        else:
            os.mkdir(self.confusion_matrices)

        # Create SummaryWriter() object
        experiment_starting_date = str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
        self.summary = SummaryWriter(log_dir=self.runpath, comment=datetime)

        # Write starting message to tensorboard.
        message = "Starting experiment '{exp_name}' with parameters {params} at time {date}".format(exp_name=self.exp_name, params=str(args), date=experiment_starting_date)
        self.add_text('Start', text=message)

    def flush(self):
        """
        Flush method (ensure that any change is written to disk when process finishes.
        """

        # Write final message to tensorboard to indicate the time of finalization
        current_datetime = str(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
        final_message = "Finishing experiment '{}' at time {}".format(self.exp_name, current_datetime)
        self.add_text(tag='Finish', text=final_message, epoch=self.args.epochs)

        self.summary.flush()

    def add_scalar(self, tag, scalar, epoch, kfold=False, k=0):
        """
        Write scalar to tensorboard summary. If kfold is set to True, then tag must be modified to include the fold
        which being evaluated.

        :param tag: the tag to include in tensorboard. It is hierarchically included (loss/train, loss/test)
        :param scalar: the value of the tag which is being included
        :param epoch: epoch of training
        :param kfold: if set, modify the tag parameter. It indicates that k-fold cross validation is being used.
        :param k: K partition number (integer)
        """

        # Modify tag if Kfold is set to true
        if kfold:
            tag = str(k) + "-fold/" + tag

        # Write registry to tensorboard
        self.summary.add_scalar(tag=tag, scalar_value=scalar, global_step=epoch)

    def add_scalars(self, tag, tag_scalar_dict, epoch, kfold=False, k=0):
        """
        Write several scalars at once. This generates a graph which allow the user to compare several metrics.

        :param tag: main tag (section of tensorboard). The name of the graph.
        :param tag_scalar_dict: dictionary containing pairs of {tag: scalar_value}.
        :param epoch: epoch of training
        :param kfold: if set, modify the tag parameter. It indicates that k-fold cross validation is being used.
        :param k: K partition number (integer)
        """

        # Modify tag if Kfold is set to true
        if kfold:
            tag = str(k) + "-fold/" + tag

        # Write data to tensorboard
        self.summary.add_scalars(main_tag=tag, tag_scalar_dict=tag_scalar_dict, global_step=epoch)

    def add_text(self, tag, text, epoch=0, kfold=False, k=0):
        """
        Write some text to tensorboard.

        :param tag: the tag to associated to the text
        :param text: the text to write to tensorboard
        :param epoch: epoch of training (default 0, useless)
        :param kfold: if set, modify the tag parameter. It indicates that k-fold cross validation is being used.
        :param k: K partition number (integer)
        """

        # Modify tag if Kfold is set to true
        if kfold:
            tag = str(k) + "-fold/" + tag

        # Write text
        self.summary.add_text(tag=tag, text_string=text, global_step=epoch)

    def add_graph(self, model, input=None):
        """
        Add graph (architecture of the neural network) to tensorboard.

        :param model: neural network (nn.Module)
        :param input: useless (ignore this)
        """

        # Write NN architecture to tensorboard
        self.summary.add_graph(model=model, input_to_model=input)

    def add_pointcloud(self, points, colors):
        """
        Write a 3D point cloud to tensorboard. This allows the researcher to easily explore the results of the
        experiments with no need to transform the point cloud to a specific format and directly see this using
        Three.js JavaScript lib.

        :param points: numpy array containing points (N, 3)
        :param colors: numpy array containing colors (N, 3). [0,1] if float, [0, 255] if integer.
        """
        # TODO. Not implemented yet.
        pass

    def save_confusion_matrix(self, confusion_matrix, epoch, k, mode):
        """
        Save a confusion matrix in CSV format.

        :param k: cross validation partition
        :param epoch: training epoch
        :param confusion_matrix: numpy array containing a confusion matrix
        :param mode: training or validation
        """

        # Path where the matrix will be saved
        matrix_path = os.path.join(self.confusion_matrices, '{mode}-k{k}-{epoch}.csv'.format(
            mode=mode,
            epoch=epoch,
            k=k
        ))

        # Store matrix in csv format
        np.savetxt(matrix_path, confusion_matrix, fmt='%d', delimiter=',')
