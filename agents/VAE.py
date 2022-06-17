from torch.utils.tensorboard import SummaryWriter
import numpy as np
from agents.save.Checkpoint import Checkpoint
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
import agents.math_fc.functions as mathfc
from singletons.Device import Device
from torch import zeros_like
from agents.learning import Optimizers
import torch


#
# Implement an agent acting randomly.
#
class VAE:

    def __init__(
            self, encoder, decoder, n_steps_beta_reset, beta, lr, beta_starting_step,
            beta_rate, queue_capacity, action_selection, tensorboard_dir, steps_done=0, **_
    ):
        """
        Constructor
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param n_steps_beta_reset: the number of steps after with beta is reset
        :param beta_starting_step: the number of steps after which beta start increasing
        :param beta: the initial value for beta
        :param beta_rate: the rate at which the beta parameter is increased
        :param lr: the learning rate
        :param queue_capacity: the maximum capacity of the queue
        :param action_selection: the action selection to be used
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param steps_done: the number of training iterations performed to date.
        """

        # Neural networks.
        self.encoder = encoder
        self.decoder = decoder

        # Ensure models are on the right device.
        Device.send([self.encoder, self.decoder])

        # Optimizer.
        self.optimizer = Optimizers.get_adam([encoder, decoder], lr)

        # Beta scheduling.
        self.n_steps_beta_reset = n_steps_beta_reset
        self.beta_starting_step = beta_starting_step
        self.beta = beta
        self.beta_rate = beta_rate

        # Miscellaneous.
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.writer = SummaryWriter(tensorboard_dir)
        self.lr = lr
        self.queue_capacity = queue_capacity
        self.tensorboard_dir = tensorboard_dir
        self.action_selection = action_selection
        self.total_rewards = 0

    def step(self, obs, config):
        """
        Select a random action
        :param obs: unused
        :param config: unused
        :return: the action to be performed
        """
        quality = torch.zeros([1, config["env"]["n_actions"]]).to(Device.get())
        return self.action_selection.select(quality, self.steps_done)

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
                self.save(config)

            # Render the environment.
            if config["enable_tensorboard"]:
                self.total_rewards += reward
                self.writer.add_scalar("Rewards", self.total_rewards, self.steps_done)
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
        _, _, _, _, next_obs = self.buffer.sample(config["batch_size"])

        # Compute the variational free energy.
        vfe_loss = self.compute_vfe(config, next_obs)

        # Perform one step of gradient descent on the other networks.
        self.optimizer.zero_grad()
        vfe_loss.backward()
        self.optimizer.step()

        # Implement the cyclical scheduling for beta.
        if self.steps_done >= self.beta_starting_step:
            self.beta = np.clip(self.beta + self.beta_rate, 0, 1)
        if self.steps_done % self.n_steps_beta_reset == 0:
            self.beta = 0

    def compute_vfe(self, config, next_obs):
        """
        Compute the variational free energy
        :param config: the hydra configuration
        :param next_obs: the observations at time t + 1
        :return: the variational free energy
        """

        # Compute required vectors.
        mean_hat, log_var_hat = self.encoder(next_obs)
        next_state = mathfc.reparameterize(mean_hat, log_var_hat)
        mean = zeros_like(next_state)
        log_var = zeros_like(next_state)
        alpha = self.decoder(next_state)

        # Compute the variational free energy.
        kl_div_hs = mathfc.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood = mathfc.log_bernoulli_with_logits(next_obs, alpha)
        vfe_loss = self.beta * kl_div_hs - log_likelihood

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("kl_div_hs", kl_div_hs, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood", - log_likelihood, self.steps_done)
            self.writer.add_scalar("beta", self.beta, self.steps_done)
            self.writer.add_scalar("vfe", vfe_loss, self.steps_done)

        return vfe_loss

    def predict(self, obs):
        """
        Do one forward pass using given observation.
        :return: the outputs of the encoder
        """
        return self.encoder(obs)

    def save(self, config):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param config: the hydra configuration.
        :return: nothing
        """

        # Create directories and files if they do not exist.
        checkpoint_file = config["checkpoint"]["file"]
        Checkpoint.create_dir_and_file(checkpoint_file)

        # Save the model.
        torch.save({
            "agent_module": str(self.__module__),
            "agent_class": str(self.__class__.__name__),
            "images_shape": config["images"]["shape"],
            "n_states": config["agent"]["n_states"],
            "decoder_net_state_dict": self.decoder.state_dict(),
            "decoder_net_module": str(self.decoder.__module__),
            "decoder_net_class": str(self.decoder.__class__.__name__),
            "encoder_net_state_dict": self.encoder.state_dict(),
            "encoder_net_module": str(self.encoder.__module__),
            "encoder_net_class": str(self.encoder.__class__.__name__),
            "action_selection": dict(self.action_selection),
            "lr": self.lr,
            "beta": self.beta,
            "n_steps_beta_reset": self.n_steps_beta_reset,
            "beta_starting_step": self.beta_starting_step,
            "beta_rate": self.beta_rate,
            "steps_done": self.steps_done,
            "queue_capacity": self.queue_capacity,
            "tensorboard_dir": self.tensorboard_dir,
        }, checkpoint_file)

    @staticmethod
    def load_constructor_parameters(tb_dir, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint.
        :param tb_dir: the path of tensorboard directory.
        :param checkpoint: the checkpoint from which to load the parameters.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: a dictionary containing the constructor's parameters.
        """
        return {
            "encoder": Checkpoint.load_encoder(checkpoint, training_mode),
            "decoder": Checkpoint.load_decoder(checkpoint, training_mode),
            "lr": checkpoint["lr"],
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "beta": checkpoint["beta"],
            "n_steps_beta_reset": checkpoint["n_steps_beta_reset"],
            "beta_starting_step": checkpoint["beta_starting_step"],
            "beta_rate": checkpoint["beta_rate"],
            "steps_done": checkpoint["steps_done"],
            "queue_capacity": checkpoint["queue_capacity"],
            "tensorboard_dir": tb_dir,
        }
