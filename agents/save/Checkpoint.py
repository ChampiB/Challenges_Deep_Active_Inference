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

    def __init__(self, tb_dir, file):
        """
        Construct the checkpoint from the checkpoint file
        :param tb_dir: the path of tensorboard directory
        :param file: the checkpoint file
        """

        # If the path is not a file, return without trying to load the checkpoint.
        if not os.path.isfile(file):
            Logger.get().warn("Could not load model from: " + file)
            self.checkpoint = None
            return

        # Load checkpoint from path.
        self.checkpoint = torch.load(file, map_location=Device.get())

        # Store the path of the tensorboard directory and the model name
        self.tb_dir = tb_dir

    def exists(self):
        """
        Check whether the checkpoint file exists.
        :return: True if the checkpoint file exists, False otherwise.
        """
        return self.checkpoint is not None

    def load_model(self, training_mode=True, override=None):
        """
        Load the model from the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :param override: the key-value pari that needs to be overridden in the checkpoint
        :return: the loaded model or None if an error occurred.
        """

        # Check if the checkpoint is loadable.
        if not self.exists():
            return None

        # Override agent module and class if needed.
        if override is not None:
            for key, value in override.items():
                self.checkpoint[key] = value

        # Load the agent class and module.
        agent_module = importlib.import_module(self.checkpoint["agent_module"])
        agent_class = getattr(agent_module, self.checkpoint["agent_class"])

        # Load the parameters of the constructor from the checkpoint.
        param = agent_class.load_constructor_parameters(self.tb_dir, self.checkpoint, training_mode)

        # Instantiate the agent.
        return agent_class(**param)

    @staticmethod
    def create_dir_and_file(checkpoint_file):
        """
        Create the directory and file of the checkpoint if they do not already exist
        :param checkpoint_file: the checkpoint file
        :return: nothing
        """
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            file = Path(checkpoint_file)
            file.touch(exist_ok=True)

    @staticmethod
    def set_training_mode(neural_net, training_mode):
        """
        Set the training mode of the neural network sent as parameters
        :param neural_net: the neural network whose training mode needs to be set
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: nothing
        """
        if training_mode:
            neural_net.train()
        else:
            neural_net.eval()

    @staticmethod
    def load_decoder(checkpoint, training_mode=True):
        """
        Load the decoder from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the decoder
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
        Load the encoder from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the encoder
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
    def load_encoder_2ls(checkpoint, training_mode=True):
        """
        Load the encoder from the checkpoint
        :param checkpoint: the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the encoder.
        """

        # Load encoder network.
        encoder_module = importlib.import_module(checkpoint["encoder_net_module"])
        encoder_class = getattr(encoder_module, checkpoint["encoder_net_class"])
        encoder = encoder_class(
            n_reward_states=checkpoint["n_reward_states"],
            n_model_states=checkpoint["n_states"],
            image_shape=checkpoint["images_shape"]
        )
        encoder.load_state_dict(checkpoint["encoder_net_state_dict"])

        # Set the training mode of the encoder.
        Checkpoint.set_training_mode(encoder, training_mode)
        return encoder

    @staticmethod
    def load_transition(checkpoint, training_mode=True):
        """
        Load the transition from the checkpoint
        :param checkpoint: the checkpoint
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :return: the transition
        """

        # Load transition network.
        transition_module = importlib.import_module(checkpoint["transition_net_module"])
        transition_class = getattr(transition_module, checkpoint["transition_net_class"])
        transition = transition_class(
            n_states=checkpoint["n_states"], n_actions=checkpoint["n_actions"]
        )
        transition.load_state_dict(checkpoint["transition_net_state_dict"])

        # Set the training mode of the transition.
        Checkpoint.set_training_mode(transition, training_mode)
        return transition

    @staticmethod
    def load_critic(checkpoint, training_mode=True, n_states_key="n_states", network_key="critic_net"):
        """
        Load the critic from the checkpoint
        :param checkpoint: the checkpoint
        :param n_states_key: the key of the dictionary containing the number of states
        :param training_mode: True if the agent is being loaded for training, False otherwise
        :param network_key: the prefix of the keys containing the critic's module and class
        :return: the critic.
        """
        # Check validity of inputs
        if network_key + '_module' not in checkpoint.keys() or network_key + '_class' not in checkpoint.keys():
            return None

        # Load critic network.
        critic_module = importlib.import_module(checkpoint[network_key + "_module"])
        critic_class = getattr(critic_module, checkpoint[network_key + "_class"])
        critic = critic_class(
            n_states=checkpoint[n_states_key], n_actions=checkpoint["n_actions"]
        )
        critic.load_state_dict(checkpoint[network_key + "_state_dict"])

        # Set the training mode of the critic.
        Checkpoint.set_training_mode(critic, training_mode)
        return critic

    @staticmethod
    def load_policy(checkpoint, training_mode):
        """
        Load the policy from the checkpoint
        :param checkpoint: the checkpoint.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: the policy.
        """
        # Load critic network.
        policy_module = importlib.import_module(checkpoint["policy_net_module"])
        policy_class = getattr(policy_module, checkpoint["policy_net_class"])
        policy = policy_class(
            images_shape=checkpoint["images_shape"], n_actions=checkpoint["n_actions"]
        )
        policy.load_state_dict(checkpoint["policy_net_state_dict"])

        # Set the training mode of the critic.
        Checkpoint.set_training_mode(policy, training_mode)
        return policy

    @staticmethod
    def load_object_from_dictionary(checkpoint, key):
        """
        Load the action selection strategy from the checkpoint
        :param checkpoint: the checkpoint
        :param key: the key in the dictionary where the object has been serialized
        :return: the action selection strategy
        """

        # Load the action selection strategy from the checkpoint.
        action_selection = checkpoint[key]
        action_selection_module = importlib.import_module(action_selection["module"])
        action_selection_class = getattr(action_selection_module, action_selection["class"])
        return action_selection_class(**action_selection)
