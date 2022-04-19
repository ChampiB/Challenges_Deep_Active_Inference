from datetime import datetime
import numpy as np
import os
import math
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, eye, zeros, softmax
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from agents.dgn_networks.EncoderNetworks import ConvEncoder64 as Encoder
from agents.dgn_networks.DecoderNetworks import ConvDecoder64 as Decoder
from agents.dgn_networks.TransitionNetworks import LinearRelu as Transition
from agents.dgn_networks.PolicyNetworks import PolicyNetwork as Policy
from agents.dgn_networks.CriticNetworks import LinearRelu as Critic
from agents.planning.PMCTS import PMCTS
from agents.planning.NodePMCTS import NodePMCTS as Node
from singletons.Device import Device

# TODO refacto this file



#
# The class implementing a Deep-G-Network with Monte Carlo Tree Search.
#
class DGN:

    #
    # Constructor.
    #

    def __init__(self, env_name, image_shape, n_actions):
        """
        Constructor.
        :param env_name: the name of the environment on which the agent is trained.
        :param image_shape: the shape of the input images.
        :param n_actions: the number of actions available to the agent.
        """

        # The device on which the code should be run.
        self.device = Device.get()

        # Create a dictionary containing the hyper-parameters.
        self.hp = {
            "saving_directory": "./data/agent_checkpoints/DGN_checkpoint_10_" + env_name + ".pt",
            "checkpoint_frequency": 100,  # Frequency at which the agent should be saved.
            "debug_mode": True,  # Should the simulation display debugging information?
            "n_latent_dims": 10,  # Number of dimensions in the latent space.
            "n_actions": n_actions,  # Number of actions.
            "max_planning_steps": 100,  # Maximum number of planning iterations.
            "n_steps_between_synchro": 10,  # Synchronization of the target with the critic.
            "n_training_steps": 10000000,  # Number of steps to run in the environment.
            "tree_search": False,  # Should the agent perform MCTS?
            "zeta": 5,  # Exploration constant of the MCTS algorithm.
            "gamma": 1,  # Precision of the prior over action P(a|s).
            "discount_factor": 0.9,  # Discount factor.
            "phi": 1,  # Precision of action selection.
            "queue_capacity": 10000,  # Replay buffer maximum capacity.
            "batch_size": 50,  # Size of batches sampled from the replay buffer.
            "buffer_start_size": 50,  # Size of the buffer at which learning start.
            "lr_efe": 0.001,  # Learning rate of the critic network.
            "lr_vfe": 0.001,  # Learning rate of all other networks.
            "beta": 0.0,  # The beta parameter of the Variational Auto-Encoder.
            "beta_starting_step": 0,  # The step at which beta starts increasing.
            "beta_rate": 0.0002,  # Rate at which beta increases during learning.
            "n_steps_beta_reset": 10000,  # Number of steps before to reset beta to zero.
        }

        # Create the agent's networks.
        self.encoder = Encoder(self.hp["n_latent_dims"], image_shape)
        self.decoder = Decoder(self.hp["n_latent_dims"], image_shape)
        self.decoder.build(self.encoder.conv_output_shape())
        self.transition = Transition(self.hp["n_latent_dims"], self.hp["n_actions"])
        self.policy = Policy(self.hp["n_latent_dims"], self.hp["n_actions"])
        self.critic = Critic(self.hp["n_latent_dims"], self.hp["n_actions"])
        self.target = Critic(self.hp["n_latent_dims"], self.hp["n_actions"])
        self.synchronize_target()
        self.networks_to_device()

        # Create the agent's optimizers.
        vfe_parameters = \
            list(self.decoder.parameters()) + \
            list(self.encoder.parameters()) + \
            list(self.transition.parameters()) + \
            list(self.policy.parameters())
        self.vfe_optimizer = Adam(vfe_parameters, lr=self.hp["lr_vfe"])

        efe_parameters = \
            list(self.critic.parameters())
        self.efe_optimizer = Adam(efe_parameters, lr=self.hp["lr_efe"])

        # Create the summary writer for monitoring with TensorBoard.
        self.writer = SummaryWriter('./data/runs/DGN')

        # Create a dictionary containing debug information.
        self.debug = {
            "vfe_loss":       torch.tensor([0.0]).to(self.device),
            "kl_div_hs":      torch.tensor([0.0]).to(self.device),
            "kl_div_action":  torch.tensor([0.0]).to(self.device),
            "log_likelihood": torch.tensor([0.0]).to(self.device),
            "efe_loss":       torch.tensor([0.0]).to(self.device),
            "immediate_efe":  torch.tensor([0.0]).to(self.device),
            "future_efe":     torch.tensor([0.0]).to(self.device),
            "total_rewards":  torch.tensor([0.0]).to(self.device),
        }

        # Create the replay buffer.
        self.buffer = ReplayBuffer(capacity=self.hp["queue_capacity"])

        # Number of training steps performed to date.
        self.steps_done = 0

        # Create the Monte Carlo Tree Search algorithm.
        self.mcts = PMCTS(self.hp)

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

        # Generate a typical trajectory and render the environment (if needed).
        if self.hp["debug_mode"]:
            env.render()

        # Train the agent.
        print("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < self.hp["n_training_steps"]:

            # Select an action.
            action = self.step(obs)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training (if needed).
            if len(self.buffer) >= self.hp["buffer_start_size"]:
                self.learn(self.steps_done)

            # Save the agent (if needed).
            if self.steps_done % self.hp["checkpoint_frequency"] == 0:
                self.save()

            # Synchronize the target with the critic (if needed).
            if self.steps_done % self.hp["n_steps_between_synchro"] == 0:
                self.synchronize_target()

            # Render the environment and monitor total rewards (if needed).
            if self.hp["debug_mode"]:
                self.debug["total_rewards"] += reward
                self.writer.add_scalar("Rewards", self.debug["total_rewards"], self.steps_done)
                env.render()

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of steps done.
            self.steps_done += 1

        # Close the environment.
        env.close()

    def step(self, obs):
        """
        Implement one action-perception cycle, i.e. inference of the latent state + planning + action selection.
        :param obs: the current observation.
        :return: the action to perform.
        """

        #
        # Perform sophisticated planning if requested by the user.
        #
        if self.hp["tree_search"]:

            # Inference.
            mean, _ = self.encoder(torch.unsqueeze(obs, dim=0))

            # Reset MCTS algorithm
            pi = softmax(self.policy(mean), dim=1)
            cost = self.critic(mean)
            self.mcts.reset(Node(mean, cost, pi))

            # Planning.
            for i in range(self.hp["max_planning_steps"]):
                s_node = self.mcts.select_node()
                e_node = self.mcts.expand_and_evaluate(s_node, self.transition, self.critic, self.policy)
                self.mcts.back_propagate(e_node)

            # Select the action to perform in the environment.
            return self.mcts.select_action()

        #
        # Otherwise, perform naive planning.
        #

        # Inference.
        mean_hat, log_var_hat = self.encoder(torch.unsqueeze(obs, dim=0))
        states = self.reparameterize(mean_hat, log_var_hat)

        # Planning as inference.
        actions_prob = softmax(self.policy(states), dim=1)

        # Action selection.
        return Categorical(actions_prob).sample()

    def learn(self, step):
        """
        Learn the parameters of the agent.
        :param step: the current training step.
        :return: nothing.
        """

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(self.hp["batch_size"])

        # Compute the expected free energy loss.
        efe_loss = self.compute_efe_loss(obs, actions, next_obs, done, rewards)

        # Perform one step of gradient descent on the critic network.
        if not math.isnan(efe_loss):
            self.efe_optimizer.zero_grad()
            efe_loss.backward()
            self.efe_optimizer.step()
        else:
            print("Warning: efe_loss is NaN (not a number).")

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(obs, actions, next_obs)

        # Perform one step of gradient descent on the other networks.
        if not math.isnan(vfe_loss):
            self.vfe_optimizer.zero_grad()
            vfe_loss.backward()
            self.vfe_optimizer.step()
        else:
            print("Warning: vfe_loss is NaN (not a number).")

        # Implement the cyclical scheduling for beta.
        if step >= self.hp["beta_starting_step"]:
            self.hp["beta"] = np.clip(self.hp["beta"] + self.hp["beta_rate"], 0, 1)
        if step % self.hp["n_steps_beta_reset"] == 0:
            self.hp["beta"] = 0

        # Print debug information.
        if self.hp["debug_mode"] and step % 10 == 0:
            self.print_debug_info(step, obs[0])

    def print_debug_info(self, step, obs):
        """
        Display useful debug information, such current variational free energy and critic's loss.
        :param step: the current training step.
        :param obs: the observations used to generate a trajectory.
        :return: nothing.
        """
        # Print the Variational Free Energy.
        self.writer.add_scalar("VFE", self.debug["vfe_loss"].item(), step)
        self.writer.add_scalar("neg_log_likelihood", -self.debug["log_likelihood"].item(), step)
        self.writer.add_scalar("KL_div_hs", self.debug["kl_div_hs"].item(), step)
        self.writer.add_scalar("KL_div_act", self.debug["kl_div_action"].item(), step)
        self.writer.add_scalar("Beta", self.hp["beta"], step)

        # Print the Expected Free Energy loss.
        self.writer.add_scalar("EFE loss", self.debug["efe_loss"].item(), step)
        self.writer.add_scalar("Discount factor", self.hp["discount_factor"], step)

        # Print a imaginary trajectory.
        images = self.imagine_trajectory(obs)
        images = torch.stack(images, dim=0)
        self.writer.add_images("An example of imagined trajectory", images)

    def imagine_trajectory(self, obs, length=5, actions=None):
        """
        Imagine a trajectory from the initial observation "obs".
        If actions is provided, the agent perform the provided sequence of actions.
        Otherwise it performs a random sequence of actions of size "length"
        :param obs: the initial observation to imagine from.
        :param length: the length of the trajectory to generate.
        :param actions: the sequence of action that needs to be imagined.
        :return: the list of generated/imagined observations.
        """

        # The imagined trajectory.
        trajectory = []

        # Create a random sequence of actions if needed.
        if actions is None:
            actions = [np.random.choice(self.hp["n_actions"]) for _ in range(0, length)]
        actions = [torch.unsqueeze(torch.tensor(action), dim=0) for action in actions]

        # Compute the initial state.
        mean, log_var = self.encoder(torch.unsqueeze(obs, dim=0))
        state = self.reparameterize(mean, log_var).detach()

        # Add a generated observation to the trajectory.
        trajectory.append(self.decoder(state)[0])

        # Imagine the consequences of the actions.
        for action in actions:

            # Sample possible future state.
            mean, log_var = self.transition(state, action)
            state = self.reparameterize(mean, log_var)

            # Add a generated observation to the trajectory.
            trajectory.append(self.decoder(state)[0])

        return trajectory

    def synchronize_target(self):
        """
        Synchronize the target with the critic.
        :return: nothing.
        """
        self.target.load_state_dict(self.critic.state_dict())
        self.target.eval()

    def networks_to_device(self):
        """save synonym
        Move all networks to the proper device.
        :return: nothing.
        """
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.transition.to(self.device)
        self.policy.to(self.device)
        self.critic.to(self.device)
        self.target.to(self.device)

    #
    # Math related functions.
    #

    def compute_efe_loss(self, obs, actions, next_obs, done, rewards):
        """
        Compute the expected free energy loss.
        :param obs: the observations at time t.
        :param actions: the actions at time t.
        :param next_obs: the observations at time t + 1.
        :param done: did the simulation ended at time t + 1 after performing the actions at time t?
        :param rewards: the rewards at time t + 1.
        :return: expected free energy loss
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = self.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        next_state = self.reparameterize(mean, log_var)
        mean_hat, log_var_hat = self.encoder(next_obs)

        # Compute the expected free energy at time t.
        efe_time_t = - rewards + self.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var, sum_dims=1)
        self.debug["immediate_efe"] = efe_time_t.to(torch.float32)

        # Compute the future expected free energy.
        future_efe = self.target(next_state).to(torch.float32).max(dim=1)[0]
        self.debug["future_efe"] = future_efe * torch.logical_not(done).to(torch.int64)

        # Compute the Expected Free Energy.
        self.debug["efe"] = self.debug["immediate_efe"] + self.hp["discount_factor"] * self.debug["future_efe"]

        # Compute the prediction of the critic network.
        critic_pred = self.critic(states)
        critic_pred = critic_pred.gather(dim=1, index=torch.unsqueeze(actions.to(torch.int64), dim=1))
        critic_pred = torch.squeeze(critic_pred)

        # Compute the expected free energy loss.
        efe_loss = nn.SmoothL1Loss()
        self.debug["efe_loss"] = efe_loss(critic_pred, self.debug["efe"].detach())
        return self.debug["efe_loss"]

    def compute_vfe(self, obs, actions, next_obs):
        """
        Compute the variational free energy.
        :param obs: the observations at time t.
        :param actions: the actions at time t.
        :param next_obs: the observations at time t + 1.
        :return: the variational free energy.
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = self.reparameterize(mean_hat, log_var_hat)
        mean_hat, log_var_hat = self.encoder(next_obs)
        next_state = self.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        alpha = self.decoder(next_state)
        pi = - self.hp["gamma"] * self.critic(states)
        pi_hat = self.policy(states)

        # Compute the variational free energy.
        self.debug["kl_div_hs"] = self.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        self.debug["kl_div_action"] = self.kl_div_categorical(pi_hat, pi.detach())
        self.debug["log_likelihood"] = self.log_bernoulli(next_obs, alpha)
        self.debug["vfe_loss"] = self.hp["beta"] * self.debug["kl_div_hs"] \
            + self.debug["kl_div_action"] - self.debug["log_likelihood"]
        return self.debug["vfe_loss"]

    @staticmethod
    def kl_div_categorical(p_hat, p):
        """
        Compute the KL-divergence between two Categorical distributions.
        :param p_hat: the log parameters of the first Categorical distribution.
        :param p: the log parameters of the second Categorical distribution.
        :return: the KL-divergence between the two Categorical distributions.
        """
        return (p_hat.exp() * (p_hat - p)).sum()

    @staticmethod
    def kl_div_gaussian(mean_hat, log_var_hat, mean, log_var, sum_dims=None):
        """
        Compute the KL-divergence between two Gaussian distributions.
        :param mean_hat: the mean of the second Gaussian distribution.
        :param log_var_hat: the log of variance of the second Gaussian distribution.
        :param mean: the mean of the first Gaussian distribution.
        :param log_var: the log of variance of the first Gaussian distribution.
        :param sum_dims: the dimensions along which to sum over before to return, by default all of them.
        :return: the KL-divergence between the two Gaussian distributions.
        """
        var = log_var.exp()
        var_hat = log_var_hat.exp()
        kl_div = log_var - log_var_hat + (mean_hat - mean) ** 2 / var
        kl_div += var_hat / var
        if sum_dims is None:
            return 0.5 * kl_div.sum()
        else:
            return 0.5 * kl_div.sum(dim=sum_dims)

    @staticmethod
    def log_bernoulli(obs, alpha):
        """
        Compute the log of probability of the observation assuming a Bernoulli distribution.
        :param obs: the observation.
        :param alpha: the parameter of the Bernoulli distribution over observation.
        :return: the log probability of the observation.
        """
        return -nn.BCELoss(reduction='sum')(alpha, obs).sum()

    @staticmethod
    def log_gaussian(rewards, mean, log_var):
        """
        Compute the log of probability of the observation assuming a Bernoulli distribution.
        :param rewards: the rewards.
        :param mean: the mean of the Gaussian distribution.
        :param log_var: the logarithm of the variance of the Gaussian distribution.
        :return: the log probability of the rewards.
        """
        return -0.5 * (log_var + (rewards - mean) ** 2 / log_var.exp()).sum()

    def reparameterize(self, mean, log_var):
        """
        Perform the reparameterization trick.
        :param mean: the mean of the Gaussian.
        :param log_var: the log of the variance of the Gaussian.
        :return: a sample from the Gaussian on which back-propagation can be performed.
        """
        nb_states = self.hp["n_latent_dims"]
        epsilon = MultivariateNormal(zeros(nb_states), eye(nb_states)).sample([mean.shape[0]])
        return epsilon * log_var.exp() + mean

    #
    # Save and reload the model.
    #

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
            "vfe_optimizer_state_dict": self.vfe_optimizer.state_dict(),
            "efe_optimizer_state_dict": self.efe_optimizer.state_dict(),
            "transition_net_state_dict": self.transition.state_dict(),
            "decoder_net_state_dict": self.decoder.state_dict(),
            "encoder_net_state_dict": self.encoder.state_dict(),
            "policy_net_state_dict": self.policy.state_dict(),
            "critic_net_state_dict": self.critic.state_dict(),
            "beta": self.hp["beta"],
            "n_steps_done": self.steps_done,
            "total_rewards": self.debug["total_rewards"]
        }, path)

    def load(self, training_mode=True):
        """
        Load the agent from a previously created checkpoint.
        :param training_mode: should the agent be loaded for training?
        :return: nothing.
        """

        # If the path is not a file, return without trying to load the model.
        if not os.path.isfile(self.hp["saving_directory"]):
            return

        # Load checkpoint from path.
        checkpoint = torch.load(self.hp["saving_directory"])

        # Load optimizers.
        self.vfe_optimizer.load_state_dict(checkpoint["vfe_optimizer_state_dict"])
        self.efe_optimizer.load_state_dict(checkpoint["efe_optimizer_state_dict"])

        # Load networks' weights.
        self.transition.load_state_dict(checkpoint["transition_net_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_net_state_dict"])
        self.encoder.load_state_dict(checkpoint["encoder_net_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_net_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_net_state_dict"])

        # Set the mode requested be the user, i.e. training or testing mode.
        if training_mode:
            self.transition.train()
            self.decoder.train()
            self.encoder.train()
            self.critic.train()
            self.policy.train()
        else:
            self.transition.eval()
            self.decoder.eval()
            self.encoder.eval()
            self.critic.eval()
            self.policy.eval()

        # Load beta.
        self.hp["beta"] = checkpoint["beta"]

        # Load number of training steps performed to date.
        self.steps_done = checkpoint["n_steps_done"]

        # Load the total amount of rewards received to date
        self.debug["total_rewards"] = checkpoint["total_rewards"]
