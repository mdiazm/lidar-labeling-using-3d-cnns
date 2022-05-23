"""
NeuralHandler class. This class is responsible from: storing & loading models, also with configurations in yml format.
"""

import os
import yaml
import torch
from models import dgcnn

class NeuralHandler:
    """
    Class responsible from storing & loading trained models, also for transfer learning.
    """

    def __init__(self, ckp_path):
        """
        Constructor
        :param ckp_path: path of the checkpoints to use
        """

        self.ckp_path = ckp_path

    def store_model(self, network: dgcnn.DGCNN, name, args):
        """
        Store a given model into the given path with .pt format. The model will be saved on checkpoints/name.pt

        :param name: Name of the model to store
        :param network: model object to store.
        """

        # Retrieve configuration of the network
        model_config = args

        # First, store configuration of the model
        config_path = os.path.join(self.ckp_path, name + '.yml')
        with open(config_path, 'w') as config_file:
            yaml.dump(model_config, config_file, default_flow_style=True)

        # After all, store weights of the network
        state_dict = network.state_dict()
        network_path = os.path.join(self.ckp_path, name + '.pt')
        torch.save(state_dict, network_path)

    def load_model(self, name) -> dgcnn.DGCNN:
        """
        Restore DGCNN model from the path checkpoints/name.pt and the corresponding parameters in YML file

        :param name: the name of the model to restore.
        :return: DGCNN model ready to use in inference
        """

        # Paths
        config_path = os.path.join(self.ckp_path, name + '.yml')
        network_path = os.path.join(self.ckp_path, name + '.pt')

        # First, retrieve config
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.UnsafeLoader)

        # After that, create network and load weights
        state_dict = torch.load(network_path)
        network: torch.nn.Module = dgcnn.DGCNN(config)
        network = torch.nn.DataParallel(network)
        network.load_state_dict(state_dict)
        network.eval()

        return network

    def load_model_transfer_learning(self, name, num_classes) -> dgcnn.DGCNN:
        """
        Restore DGCNN model from the path checkpoints/name.pt and the corresponding parameters in YML file

        :param name: the name of the model to restore.
        :return: DGCNN model ready to use in inference
        """

        # Paths
        config_path = os.path.join(self.ckp_path, name + '.yml')
        network_path = os.path.join(self.ckp_path, name + '.pt')

        # First, retrieve config
        with open(config_path, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.UnsafeLoader)

        # After that, create network and load weights
        state_dict = torch.load(network_path)
        network: dgcnn.DGCNN = dgcnn.DGCNN(config)
        network = torch.nn.DataParallel(network)
        network.load_state_dict(state_dict)
        network.module.change_number_output_classes(num_classes)

        # Adjust network to have num_classes output for transfer learning
        # if network.module:
        #     network.module.conv9.out_channels = num_classes
        #     network.module.num_classes = num_classes
        # else:
        #     network.conv9.out_channels = num_classes

        network.train()

        return network