from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn, zeros, eye
from torch.optim import Adam
import os
import torch


#
# Implement a HMM agent acting randomly.
#
class HMM:

    def __init__(self, encoder, decoder, transition, n_steps_beta_reset=10e16, beta=1, lr=0.0001,
                 beta_starting_step=0, beta_rate=0, queue_capacity=10000,
                 tensorboard_dir="./data/runs/VAE", **_):
        """
        Constructor
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param n_steps_beta_reset: the number of steps after with beta is reset
        :param beta_starting_step: the number of steps after which beta start increasing
        :param beta: the initial value for beta
        :param beta_rate: the rate at which the beta parameter is increased
        :param lr: the learning rate
        :param queue_capacity: the maximum capacity of the queue
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        """

        # Neural networks.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition

        # Optimizer.
        params = \
            list(encoder.parameters()) + \
            list(decoder.parameters()) + \
            list(transition.parameters())
        self.optimizer = Adam(params, lr=lr)

        # Beta scheduling.
        self.n_steps_beta_reset = n_steps_beta_reset
        self.beta_starting_step = beta_starting_step
        self.beta = beta
        self.beta_rate = beta_rate

        # Miscellaneous.
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = 0
        self.writer = SummaryWriter(tensorboard_dir)

    @staticmethod
    def step(_, config):
        """
        Select a random action
        :param _: unused
        :param config: the hydra configuration
        :return: the random action
        """
        return np.random.choice(config["env"]["n_actions"])

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
        if config["display_gui"]:
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

            # Render the environment.
            if config["display_gui"]:
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
        obs, action, _, _, next_obs = self.buffer.sample(config["batch_size"])

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(config, obs, action, next_obs)

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        vfe_loss.backward()
        self.optimizer.step()

        # Implement the cyclical scheduling for beta.
        if self.steps_done >= self.beta_starting_step:
            self.beta = np.clip(self.beta + self.beta_rate, 0, 1)
        if self.steps_done % self.n_steps_beta_reset == 0:
            self.beta = 0

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
        kl_div_hs = self.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood = self.log_bernoulli_with_logits(next_obs, alpha)
        vfe_loss = self.beta * kl_div_hs - log_likelihood

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("KL_div_hs", kl_div_hs, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood", - log_likelihood, self.steps_done)
            self.writer.add_scalar("Beta", self.beta, self.steps_done)
            self.writer.add_scalar("VFE", vfe_loss, self.steps_done)

        return vfe_loss

    @staticmethod
    def kl_div_gaussian(mean_hat, log_var_hat, mean, log_var, sum_dims=None):
        """
        Compute the KL-divergence between two Gaussian distributions
        :param mean_hat: the mean of the second Gaussian distribution
        :param log_var_hat: the log of variance of the second Gaussian distribution
        :param mean: the mean of the first Gaussian distribution
        :param log_var: the log of variance of the first Gaussian distribution
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
