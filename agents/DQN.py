from torch import nn, unsqueeze
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from agents.dqn_networks.PolicyNetworks import ConvPolicy
import torch
from copy import deepcopy
from pathlib import Path
from torch.optim import Adam
import math
import os
import numpy as np
import random


#
# Class implementing a Deep Q-network.
#
class DQN:

    #
    # Instances construction.
    #

    def __init__(self, env_name, image_shape, n_actions, device=None):
        """
        Constructor.
        :param env_name: the name of the environment on which the agent is trained.
        :param image_shape: the shape of the input images.
        :param n_actions: the number of actions.
        :param device: the device one which the model and tensor should be stored, i.e. GPU or CPU.
        """

        # Create a dictionary containing the hyper-parameters.
        self.hp = {
            "saving_directory": "./data/agent_checkpoints/DQN_checkpoint_10_" + env_name + ".pt",
            "checkpoint_frequency": 100,  # Frequency at which the agent should be saved.
            "image_shape": image_shape,  # Shape of the input images.
            "debug_mode": True,  # Should the simulation display debugging information?
            "n_actions": n_actions,  # Number of actions.
            "n_steps_between_synchro": 10,  # Synchronization of the target with the critic.
            "n_training_steps": 10000000,  # Number of steps to run in the environment.
            "discount_factor": 0.9,  # Discount factor.
            "queue_capacity": 10000,  # Replay buffer maximum capacity.
            "batch_size": 50,  # Size of batches sampled from the replay buffer.
            "buffer_start_size": 50,  # Size of the buffer at which learning start.
            "lr": 0.001,  # Learning rate of the policy network.
            "epsilon_start": 0.9,  # The initial value of epsilon (for epsilon-greedy).
            "epsilon_end": 0.05,  # The final value of epsilon (for epsilon-greedy).
            "epsilon_decay": 200,  # How slowly should epsilon decay? The bigger, the slower.
        }

        # The device on which the code should be run.
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter('./data/runs/DQN')

        # Number of training steps performed to date.
        self.step_done = 0

        # Create the full policy network.
        self.policy_net = ConvPolicy(self.hp["image_shape"], self.hp["n_actions"])
        self.policy_net.to(self.device)

        # Create the target network.
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()

        # Create the agent's optimizers.
        parameters = list(self.policy_net.parameters())
        self.optimizer = Adam(parameters, lr=self.hp["lr"])

        # Create the replay buffer.
        self.buffer = ReplayBuffer(capacity=self.hp["queue_capacity"], device=device)

        # Create a dictionary containing debug information.
        self.debug = {
            "loss": torch.tensor([0.0]),
            "total_rewards": torch.tensor([0.0]),
        }

    #
    # Main functionalities.
    #

    def train(self, env):
        """
        Train the agent in the gym environment passed as parameters.
        :param env: the gym environment.
        :return: nothing.
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Render the environment.
        if self.hp["debug_mode"]:
            env.render()

        # Train the agent.
        print("Start the training at {time}".format(time=datetime.now()))
        while self.step_done < self.hp["n_training_steps"]:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training if needed.
            if len(self.buffer) >= self.hp["buffer_start_size"]:
                self.learn(self.step_done)

            # Save the agent if needed.
            if self.step_done % self.hp["checkpoint_frequency"] == 0:
                self.save()

            # Render the environment if needed.
            if self.hp["debug_mode"]:
                self.debug["total_rewards"] += reward
                self.writer.add_scalar("Rewards", self.debug["total_rewards"].item(), self.step_done)
                env.render()

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of iterations performed.
            self.step_done += 1

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
        epsilon_threshold = self.hp["epsilon_end"] + \
            (self.hp["epsilon_start"] - self.hp["epsilon_end"]) * \
            math.exp(-1. * self.step_done / self.hp["epsilon_decay"])

        # Sample a number between 0 and 1, and either execute a random action or
        # the reward maximizing action according to the sampled value.
        sample = random.random()
        if sample > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(obs).max(1)[1].item()  # Best action.
        else:
            return np.random.choice(self.hp["n_actions"])  # Random action.

    def learn(self, step):
        """
        Learn the parameters of the agent.
        :param step: the current training step.
        :return: nothing.
        """

        # Synchronize the target network with the policy network if needed
        if self.step_done % self.hp["n_steps_between_synchro"] == 0:
            self.target_net = deepcopy(self.policy_net)
            self.target_net.eval()

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(self.hp["batch_size"])

        # Compute the variational free energy.
        loss = self.compute_loss(obs, actions, rewards, done, next_obs)

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Print debug information.
        if self.hp["debug_mode"] and step % 10 == 0:
            self.writer.add_scalar("Q-values loss", self.debug["loss"].item(), step)

    def compute_loss(self, obs, actions, rewards, done, next_obs):
        """
        Compute the loss function used to train the policy network.
        :param obs: the observations made at time t.
        :param actions: the actions performed at time t.
        :param rewards: the rewards received at time t + 1.
        :param done: did the episode ended after performing action a_t?
        :param next_obs: the observations made at time t + 1.
        :return:
        """

        # Compute the q-values of the current state and action
        # as predicted by the policy network, i.e. Q(s_t, a_t).
        state_action_values = self.policy_net(obs).gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))

        # For each batch entry where the simulation did not stopped, compute
        # the value of the next states, i.e. V(s_{t+1}). Those values are computed
        # using the target network.
        next_state_values = torch.zeros(self.hp["batch_size"])
        next_state_values[torch.logical_not(done)] = self.target_net(next_obs[torch.logical_not(done)]).max(1)[0]
        next_state_values = next_state_values.detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hp["discount_factor"]) + rewards

        # Compute the loss function.
        loss = nn.SmoothL1Loss()
        self.debug["loss"] = loss(state_action_values, expected_state_action_values.unsqueeze(1))
        return self.debug["loss"]

    #
    # Saving and reloading the model.
    #

    def load(self, training_mode=True):
        """
        Load the agent from a previously created checkpoint.
        :param training_mode: should the agent be loaded for training?
        :return: nothing.
        """

        # Get the path from which the agent should be reloaded.
        path = self.hp["saving_directory"]

        # If the path is not a file, return without trying to load the model.
        if not os.path.isfile(path):
            return

        # Load checkpoint from path
        checkpoint = torch.load(path)

        # Load the number of training operation done to date.
        self.step_done = checkpoint["step_done"]

        # Load optimizers
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load networks' weights
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])

        # Set the mode requested be the user, i.e. training or testing mode.
        if training_mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def save(self):
        """
        Create a checkpoint file allowing the agent to be reloaded later.
        :return: nothing.
        """

        # Get the path in which the agent should be saved.
        path = self.hp["saving_directory"]

        # Create directories and files if they does not exist.
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            file = Path(path)
            file.touch(exist_ok=True)

        # Save the model.
        torch.save({
            "step_done": self.step_done,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_net_state_dict": self.policy_net.state_dict(),
        }, path)
