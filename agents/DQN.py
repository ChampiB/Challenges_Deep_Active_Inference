from torch import nn, unsqueeze
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from agents.networks.PolicyNetworks import ConvPolicy
import torch
from copy import deepcopy
from pathlib import Path
from torch.optim import Adam
import math
import os
import numpy as np
import random
from singletons.Device import Device


#
# Class implementing a Deep Q-network.
#
class DQN:

    #
    # Instances construction.
    #

    def __init__(
            self, image_shape, n_actions, n_steps_between_synchro, discount_factor,
            queue_capacity, lr, epsilon_start, epsilon_end, epsilon_decay,
            tensorboard_dir, **_
    ):
        """
        Constructor.
        :param env_name: the name of the environment on which the agent is trained.
        :param image_shape: the shape of the input images.
        :param n_actions: the number of actions.
        :param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        :param discount_factor: the value by which future rewards are discounted
        :param queue_capacity: the maximum capacity of the queue
        :param lr: the learning rate of the Q-network
        :param epsilon_start: the initial value for main parameter of the epsilon greedy
        :param epsilon_end: the final value for main parameter of the epsilon greedy
        :param epsilon_decay: the decay at which the episilon parameter decreases
        """

        # The hyperparameters.
        self.n_actions = n_actions
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # The device on which the code should be run.
        self.device = Device.get()

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter(tensorboard_dir)

        # Number of training steps performed to date.
        self.steps_done = 0

        # Create the full policy network.
        self.policy_net = ConvPolicy(image_shape, n_actions)
        self.policy_net.to(self.device)

        # Create the target network.
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()

        # Create the agent's optimizers.
        parameters = list(self.policy_net.parameters())
        self.optimizer = Adam(parameters, lr=lr)

        # Create the replay buffer.
        self.buffer = ReplayBuffer(capacity=queue_capacity)

        # Create a dictionary containing debug information.
        self.debug = {
            "loss": torch.tensor([0.0]),
            "total_rewards": torch.tensor([0.0]),
        }

    #
    # Main functionalities.
    #

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters.
        :param env: the gym environment.
        :param config: the hydra configuration.
        :return: nothing.
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Render the environment.
        if config["display_gui"]:
            env.render()

        # Train the agent.
        print("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config["n_training_steps"]:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training if needed.
            if len(self.buffer) >= config["buffer_start_size"]:
                self.learn(config)

            # Save the agent if needed.
            if self.steps_done % config["checkpoint"]["frequency"] == 0:
                self.save(config["checkpoint"]["file"])

            # Render the environment if needed.
            if config["enable_tensorboard"]:
                self.debug["total_rewards"] += reward
                self.writer.add_scalar("Rewards", self.debug["total_rewards"].item(), self.steps_done)
            if config["display_gui"]:
                env.render()

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of iterations performed.
            self.steps_done += 1

        # Close the environment.
        env.close()

    def step(self, obs):
        """
        Choose the next action to perform in the environment.
        :param obs: the current observation.
        :return: the action to perform.
        """

        # Create a 4D tensor from a 3D tensor by adding a dimension of size one.
        obs = torch.unsqueeze(obs, dim=0)

        # Compute the current epsilon value.
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

        # Sample a number between 0 and 1, and either execute a random action or
        # the reward maximizing action according to the sampled value.
        sample = random.random()
        if sample > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(obs).max(1)[1].item()  # Best action.
        else:
            return np.random.choice(self.n_actions)  # Random action.

    def learn(self, config):
        """
        Learn the parameters of the agent.
        :param config: the hydra configuration
        :return: nothing.
        """

        # Synchronize the target network with the policy network if needed
        if self.steps_done % self.n_steps_between_synchro == 0:
            self.target_net = deepcopy(self.policy_net)
            self.target_net.eval()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(config["batch_size"])

        # Compute the policy network's loss function.
        loss = self.compute_loss(config, obs, actions, rewards, done, next_obs)

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Print debug information.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("Q-values loss", self.debug["loss"].item(), self.steps_done)

    def compute_loss(self, config, obs, actions, rewards, done, next_obs):
        """
        Compute the loss function used to train the policy network.
        :param config: the hydra configuration
        :param obs: the observations made at time t.
        :param actions: the actions performed at time t.
        :param rewards: the rewards received at time t + 1.
        :param done: did the episode ended after performing action a_t?
        :param next_obs: the observations made at time t + 1.
        :return:
        """

        # Compute the q-values of the current state and action
        # as predicted by the policy network, i.e. Q(s_t, a_t).
        policy_pred = self.policy_net(obs)\
            .gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))

        # For each batch entry where the simulation did not stop, compute
        # the value of the next states, i.e. V(s_{t+1}). Those values are computed
        # using the target network.
        future_values = torch.zeros(config["batch_size"]).to(Device.get())
        future_values[torch.logical_not(done)] = self.target_net(next_obs[torch.logical_not(done)]).max(1)[0]
        future_values = future_values.detach()

        # Compute the expected Q values
        total_values = rewards + future_values * self.discount_factor

        # Compute the loss function.
        loss = nn.SmoothL1Loss()
        self.debug["loss"] = loss(policy_pred, total_values.unsqueeze(1))
        return self.debug["loss"]

    #
    # Saving and reloading the model.
    #

    def load(self, checkpoint_directory, training_mode=True):
        """
        Load the agent from a previously created checkpoint
        :param checkpoint_directory: the directory containing the model
        :param training_mode: should the agent be loaded for training?
        :return: nothing
        """

        # If the path is not a file, return without trying to load the model.
        if not os.path.isfile(checkpoint_directory):
            return

        # Load checkpoint from path
        checkpoint = torch.load(checkpoint_directory)

        # Load the number of training operation done to date.
        self.steps_done = checkpoint["step_done"]

        # Load optimizers
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load networks' weights
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])

        # Set the mode requested be the user, i.e. training or testing mode.
        if training_mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def save(self, path):
        """
        Create a checkpoint file allowing the agent to be reloaded later.
        :param path: the path at which the agent should be saved.
        :return: nothing.
        """

        # Create directories and files if they do not exist.
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            file = Path(path)
            file.touch(exist_ok=True)

        # Save the model.
        torch.save({
            "step_done": self.steps_done,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_net_state_dict": self.policy_net.state_dict(),
        }, path)
