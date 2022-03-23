from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy
from pathlib import Path
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch import nn, zeros, eye, unsqueeze, softmax
from torch.optim import Adam
import os
import torch


#
# Implement a HMM agent acting randomly, able to evaluate
# the qualities of each action.
#
class CriticalHMM:

    def __init__(self, encoder, decoder, transition, critic, discount_factor=0.9,
                 n_steps_beta_reset=10e16, beta=1, efe_lr=0.0001, vfe_lr=0.0001,
                 beta_starting_step=0, beta_rate=0, queue_capacity=10000,
                 n_steps_between_synchro=10, tensorboard_dir="./data/runs/VAE", **_):
        """
        Constructor
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param critic: the critic network
        :param n_steps_beta_reset: the number of steps after with beta is reset
        :param beta_starting_step: the number of steps after which beta start increasing
        :param beta: the initial value for beta
        :param beta_rate: the rate at which the beta parameter is increased
        :param efe_lr: the learning rate of the critic network
        :param vfe_lr: the learning rate of the other networks
        :param queue_capacity: the maximum capacity of the queue
        :param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        """

        # Neural networks.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.critic = critic
        self.target = copy.deepcopy(critic)

        # Optimizers.
        vfe_params = \
            list(encoder.parameters()) + \
            list(decoder.parameters()) + \
            list(transition.parameters())
        self.vfe_optimizer = Adam(vfe_params, lr=vfe_lr)

        efe_params = \
            list(critic.parameters())
        self.efe_optimizer = Adam(efe_params, lr=efe_lr)

        # Beta scheduling.
        self.n_steps_beta_reset = n_steps_beta_reset
        self.beta_starting_step = beta_starting_step
        self.beta = beta
        self.beta_rate = beta_rate

        # Miscellaneous.
        self.total_rewards = 0.0
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = 0
        self.writer = SummaryWriter(tensorboard_dir)

    def step(self, obs, config):
        """
        Select a random action based on the critic ouput
        :param obs: the input observation from which decision should be made
        :param config: the hydra configuration
        :return: the random action
        """

        # Inference.
        mean_hat, log_var_hat = self.encoder(torch.unsqueeze(obs, dim=0))
        states = self.reparameterize(mean_hat, log_var_hat)

        # Planning as inference.
        actions_prob = softmax(self.critic(states), dim=1)

        # Action selection.
        return Categorical(actions_prob).sample()

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the gym environment
        :param config: the hydra configuration
        :return: nothing
        """

        # Retrieve the initial observation from the environment.
        obs = env.reset()

        # Render the environment (if needed).
        if config["debug_mode"]:
            env.render()

        # Train the agent.
        Logger.get().info("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config["n_training_steps"]:

            # Select an action.
            action = self.step(obs, config)

            # Execute the action in the environment.
            old_obs = obs
            obs, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_obs, action, reward, done, obs))

            # Perform one iteration of training (if needed).
            if len(self.buffer) >= config["buffer_start_size"]:
                self.learn(config)

            # Save the agent (if needed).
            if self.steps_done % config["checkpoint"]["frequency"] == 0:
                self.save(config["checkpoint"]["directory"])

            # Synchronize the target with the critic (if needed).
            if self.steps_done % self.n_steps_between_synchro == 0:
                self.synchronize_target()

            # Render the environment and monitor total rewards (if needed).
            if config["debug_mode"]:
                self.total_rewards += reward
                self.writer.add_scalar("Rewards", self.total_rewards, self.steps_done)
                env.render()

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Increase the number of steps done.
            self.steps_done += 1

        # Close the environment.
        env.close()

    def learn(self, config):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param config: the hydra configuration
        :return: nothing
        """

        # Sample the replay buffer.
        obs, actions, rewards, done, next_obs = self.buffer.sample(config["batch_size"])

        # Compute the expected free energy loss.
        efe_loss = self.compute_efe_loss(config, obs, actions, next_obs, done, rewards)

        # Perform one step of gradient descent on the critic network.
        self.efe_optimizer.zero_grad()
        efe_loss.backward()
        self.efe_optimizer.step()

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(config, obs, actions, next_obs)

        # Perform one step of gradient descent on the other networks.
        self.vfe_optimizer.zero_grad()
        vfe_loss.backward()
        self.vfe_optimizer.step()

        # Implement the cyclical scheduling for beta.
        if self.steps_done >= self.beta_starting_step:
            self.beta = np.clip(self.beta + self.beta_rate, 0, 1)
        if self.steps_done % self.n_steps_beta_reset == 0:
            self.beta = 0

    def compute_efe_loss(self, config, obs, actions, next_obs, done, rewards):
        """
        Compute the expected free energy loss
        :param config: the hydra configuration
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :param done: did the simulation ended at time t + 1 after performing the actions at time t
        :param rewards: the rewards at time t + 1
        :return: expected free energy loss
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = self.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        next_state = self.reparameterize(mean, log_var)
        mean_hat, log_var_hat = self.encoder(next_obs)

        # Compute the expected free energy at time t.
        immediate_efe = - rewards + self.entropy_gaussian(log_var_hat) - self.entropy_gaussian(log_var)
        immediate_efe = immediate_efe.to(torch.float32)

        # Compute the future expected free energy.
        future_efe = self.target(next_state).to(torch.float32).max(dim=1)[0]
        future_efe = future_efe * torch.logical_not(done).to(torch.int64)

        # Compute the Expected Free Energy.
        efe = immediate_efe + self.discount_factor * future_efe

        # Compute the prediction of the critic network.
        critic_pred = self.critic(states)
        critic_pred = critic_pred.gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))
        critic_pred = torch.squeeze(critic_pred)

        # Compute the expected free energy loss.
        efe_loss = nn.SmoothL1Loss()
        efe_loss = efe_loss(critic_pred, efe.detach())

        # Display debug information, if needed.
        if config["debug_mode"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("(mean) immediate efe", immediate_efe.mean(), self.steps_done)
            self.writer.add_scalar("(mean) future efe ", future_efe.mean(), self.steps_done)
            self.writer.add_scalar("discount factor", self.discount_factor, self.steps_done)
            self.writer.add_scalar("(mean) efe", efe.mean(), self.steps_done)
            self.writer.add_scalar("efe_loss", efe_loss, self.steps_done)

        return efe_loss

    def compute_vfe(self, config, obs, actions, next_obs):
        """
        Compute the variational free energy
        :param config: the hydra configuration
        :param obs: the observations at time t
        :param actions: the actions at time t
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(obs)
        states = self.reparameterize(mean_hat, log_var_hat)
        mean_hat, log_var_hat = self.encoder(next_obs)
        next_state = self.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        alpha = self.decoder(next_state)

        # Compute the variational free energy.
        kl_div_hs = self.kl_div_gaussian(mean, log_var, mean_hat, log_var_hat)
        log_likelihood = self.log_bernoulli_with_logits(next_obs, alpha)
        vfe_loss = self.beta * kl_div_hs - log_likelihood

        # Display debug information, if needed.
        if config["debug_mode"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("KL_div_hs", kl_div_hs, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood", - log_likelihood, self.steps_done)
            self.writer.add_scalar("Beta", self.beta, self.steps_done)
            self.writer.add_scalar("VFE", vfe_loss, self.steps_done)

        return vfe_loss

    @staticmethod
    def entropy_gaussian(log_var, sum_dims=None):
        """
        Compute the entropy of a Gaussian distribution.
        :param log_var: the logarithm of the variance parameter.
        :param sum_dims: the dimensions along which to sum over before to return, by default only dimension one.
        :return: the entropy of a Gaussian distribution.
        """
        ln2pie = 1.23247435026
        sum_dims = [1] if sum_dims is None else sum_dims
        return log_var.size()[1] * 0.5 * ln2pie + 0.5 * log_var.sum(sum_dims)

    @staticmethod
    def kl_div_gaussian(mean, log_var, mean_hat, log_var_hat, sum_dims=None):
        """
        Compute the KL-divergence between two Gaussian distributions
        :param mean: the mean of the first Gaussian distribution
        :param log_var: the log of variance of the first Gaussian distribution
        :param mean_hat: the mean of the second Gaussian distribution
        :param log_var_hat: the log of variance of the second Gaussian distribution
        :param sum_dims: the dimensions along which to sum over before to return, by default all of them
        :return: the KL-divergence between the two Gaussian distributions
        """
        var = log_var.exp()
        var_hat = log_var_hat.exp()
        kl_div = log_var - log_var_hat + (mean_hat - mean) ** 2 / var
        kl_div += var_hat / var

        if sum_dims is None:
            return 0.5 * kl_div.sum(dim=1).mean()
        else:
            return 0.5 * kl_div.sum(dim=sum_dims)

    @staticmethod
    def log_bernoulli(obs, alpha, using_pytorch=False):
        """
        Compute the log of probability of the observation assuming a Bernoulli distribution
        :param obs: the observation
        :param alpha: the parameter of the Bernoulli distribution over observation
        :param using_pytorch: should pytorch BCELoss be used?
        :return: the log probability of the observation
        """
        if using_pytorch:
            return -nn.BCELoss(reduction='none')(alpha, obs).mean(dim=0).sum()
        else:
            epsilon = 0.0001
            alpha = torch.clip(alpha, epsilon, 1 - epsilon)
            out = obs * torch.log(alpha) + (1 - obs) * torch.log(1 - alpha)
            return out.sum(dim=(1, 2, 3)).mean()

    @staticmethod
    def log_bernoulli_with_logits(obs, alpha):
        """
        Compute the log probabilit of the observation (obs), given the logits (alpha), assuming
        a bernoulli distribution
        :param obs: the observation
        :param alpha: the logits
        :return: the log probability of the observation
        """
        out = torch.exp(alpha)
        one = torch.ones_like(out)
        out = alpha * obs - torch.log(one + out)
        return out.sum(dim=(1, 2, 3)).mean()

    @staticmethod
    def reparameterize(mean, log_var):
        """
        Perform the reparameterization trick
        :param mean: the mean of the Gaussian
        :param log_var: the log of the variance of the Gaussian
        :return: a sample from the Gaussian on which back-propagation can be performed
        """
        nb_states = mean.shape[1]
        epsilon = MultivariateNormal(zeros(nb_states), eye(nb_states)).sample([mean.shape[0]])
        return epsilon * (0.5 * log_var).exp() + mean

    def synchronize_target(self):
        """
        Synchronize the target with the critic.
        :return: nothing.
        """
        self.target.load_state_dict(self.critic.state_dict())
        self.target.eval()

    def save(self, checkpoint_directory):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param checkpoint_directory: the directory in which to save the model
        :return: nothing
        """

        # Create directories and files if they do not exist.
        if not os.path.exists(os.path.dirname(checkpoint_directory)):
            os.makedirs(os.path.dirname(checkpoint_directory))
            file = Path(checkpoint_directory)
            file.touch(exist_ok=True)

        # Save the model.
        torch.save({
            "decoder_net_state_dict": self.decoder.state_dict(),
            "encoder_net_state_dict": self.encoder.state_dict(),
            "transition_net_state_dict": self.transition.state_dict(),
            "critic_net_state_dict": self.critic.state_dict(),
            "n_steps_beta_reset": self.n_steps_beta_reset,
            "beta_starting_step": self.beta_starting_step,
            "beta": self.beta,
            "beta_rate": self.beta_rate,
            "n_steps_done": self.steps_done,
        }, checkpoint_directory)

    def load(self, checkpoint_directory, training_mode=True):
        """
        Load the agent from a previously created checkpoint
        :param checkpoint_directory: the directory containing the model
        :param training_mode: should the agent be loaded for training?
        :return: nothing
        """

        # If the path is not a file, return without trying to load the model.
        if not os.path.isfile(checkpoint_directory):
            Logger.get().warn("Could not load model from: " + checkpoint_directory)
            return

        # Load checkpoint from path.
        checkpoint = torch.load(checkpoint_directory)

        # Load networks' weights.
        self.decoder.load_state_dict(checkpoint["decoder_net_state_dict"])
        self.encoder.load_state_dict(checkpoint["encoder_net_state_dict"])
        self.transition.load_state_dict(checkpoint["transition_net_state_dict"])

        # Set the mode requested be the user, i.e. training or testing mode.
        if training_mode:
            self.decoder.train()
            self.encoder.train()
            self.transition.train()
        else:
            self.decoder.eval()
            self.encoder.eval()
            self.transition.eval()

        # Load parameters of beta scheduling.
        self.beta = checkpoint["beta"]
        self.n_steps_beta_reset = checkpoint["n_steps_beta_reset"]
        self.beta_starting_step = checkpoint["beta_starting_step"]
        self.beta_rate = checkpoint["beta_rate"]

        # Load number of training steps performed to date.
        self.steps_done = checkpoint["n_steps_done"]
