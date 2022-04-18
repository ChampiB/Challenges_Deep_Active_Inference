import os
from pathlib import Path
from singletons.Device import Device
from singletons.Logger import Logger
import importlib
import torch


#
# Class allowing to load model checkpoints.
#
class Checkpoint:

    def __init__(self, config, file):
        """
        Construct the checkpoint from the checkpoint file.
        :param config: the hydra configuration.
        :param file: the checkpoint file.
        """

        # If the path is not a file, return without trying to load the checkpoint.
        if not os.path.isfile(file):
            Logger.get().warn("Could not load model from: " + file)
            self.checkpoint = None
            return

        # Load checkpoint from path.
        self.checkpoint = torch.load(file, map_location=Device.get())

        # Store the configuration
        self.config = config

    def exists(self):
        """
        Check whether the checkpoint file exists.
        :return: True if the checkpoint file exists, False otherwise.
        """
        return self.checkpoint is not None

    def load_model(self, training_mode=True):
        """
        Load the model from the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the loaded model or None if an error occured.
        """

        # Check if the checkpoint is loadable.
        if not self.exists():
            return None

        # Load the agent class and module.
        agent_module = importlib.import_module(self.checkpoint["agent_module"])
        agent_class = getattr(agent_module, self.checkpoint["agent_class"])

        # Load the parameters of the constructor from the checkpoint.
        param = agent_class.load_constructor_parameters(self.config, self.checkpoint, training_mode)

        # Instantiate the agent.
        return agent_class(**param)

    @staticmethod
    def create_dir_and_file(checkpoint_file):
        """
        Create the directory and file of the checkpoint if they do not already exist.
        :param checkpoint_file: the checkpoint file.
        :return: nothing.
        """
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            file = Path(checkpoint_file)
            file.touch(exist_ok=True)

    @staticmethod
    def set_training_mode(neural_net, training_mode):
        """
        Set the training mode of the neural network sent as parameters.
        :param neural_net: the neural network whose training mode needs to be set.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: nothing.
        """
        if training_mode:
            neural_net.train()
        else:
            neural_net.eval()

    @staticmethod
    def load_decoder(checkpoint, training_mode=True):
        """
        Load the decoder from the checkpoint.
        :param checkpoint: the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the decoder.
        """

        # Load number of states and the image shape.
        image_shape = checkpoint["images_shape"]
        n_states = checkpoint["n_states"]

        # Load decoder network.
        decoder_module = importlib.import_module(checkpoint["decoder_net_module"])
        decoder_class = getattr(decoder_module, checkpoint["decoder_net_class"])
        decoder = decoder_class(n_states=n_states, image_shape=image_shape)
        decoder.load_state_dict(checkpoint["decoder_net_state_dict"])

        # Set the training mode of the decoder.
        Checkpoint.set_training_mode(decoder, training_mode)
        return decoder

    @staticmethod
    def load_encoder(checkpoint, training_mode=True):
        """
        Load the encoder from the checkpoint.
        :param checkpoint: the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the encoder.
        """

        # Load number of states and the image shape.
        image_shape = checkpoint["images_shape"]
        n_states = checkpoint["n_states"]

        # Load encoder network.
        encoder_module = importlib.import_module(checkpoint["encoder_net_module"])
        encoder_class = getattr(encoder_module, checkpoint["encoder_net_class"])
        encoder = encoder_class(n_states=n_states, image_shape=image_shape)
        encoder.load_state_dict(checkpoint["encoder_net_state_dict"])

        # Set the training mode of the encoder.
        Checkpoint.set_training_mode(encoder, training_mode)
        return encoder

    @staticmethod
    def load_transition(checkpoint, training_mode=True):
        """
        Load the transition from the checkpoint.
        :param checkpoint: the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the transition.
        """

        # Load number of states and actions.
        n_actions = checkpoint["n_actions"]
        n_states = checkpoint["n_states"]

        # Load transition network.
        transition_module = importlib.import_module(checkpoint["transition_net_module"])
        transition_class = getattr(transition_module, checkpoint["transition_net_class"])
        transition = transition_class(n_states=n_states, n_actions=n_actions)
        transition.load_state_dict(checkpoint["transition_net_state_dict"])

        # Set the training mode of the transition.
        Checkpoint.set_training_mode(transition, training_mode)
        return transition

    @staticmethod
    def load_critic(checkpoint, training_mode=True):
        """
        Load the critic from the checkpoint.
        :param checkpoint: the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the critic.
        """

        # Load number of states and actions.
        n_actions = checkpoint["n_actions"]
        n_states = checkpoint["n_states"]

        # Load critic network.
        critic_module = importlib.import_module(checkpoint["critic_net_module"])
        critic_class = getattr(critic_module, checkpoint["critic_net_class"])
        critic = critic_class(n_states=n_states, n_actions=n_actions)
        critic.load_state_dict(checkpoint["critic_net_state_dict"])

        # Set the training mode of the critic.
        Checkpoint.set_training_mode(critic, training_mode)
        return critic