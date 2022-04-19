from torch.utils.tensorboard import SummaryWriter
from agents.save.Checkpoint import Checkpoint
import numpy as np
import copy
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from singletons.Device import Device
from torch.distributions.categorical import Categorical
from torch import nn, unsqueeze
from torch.optim import Adam
import agents.math_fc.functions as mathfc
import torch


#
# Implement a Critical and Adversarial HMM able to evaluate
# the qualities of each action.
#
class CAHMM:

    def __init__(
            self, encoder, decoder, transition, critic, discount_factor,
            n_steps_beta_reset, beta, efe_lr, discriminator_lr, vfe_lr,
            beta_starting_step, beta_rate, queue_capacity, n_steps_between_synchro,
            tensorboard_dir, g_value, discriminator_threshold, n_actions,
            steps_done=0, **_
    ):
        """
        Constructor
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param critic: the critic network
        :param discount_factor: the factor by which the future EFE is discounted.
        :param n_steps_beta_reset: the number of steps after with beta is reset
        :param beta_starting_step: the number of steps after which beta start increasing
        :param beta: the initial value for beta
        :param beta_rate: the rate at which the beta parameter is increased
        :param efe_lr: the learning rate of the critic network
        :param discriminator_lr: the learning rate of the discriminator network
        :param vfe_lr: the learning rate of the other networks
        :param queue_capacity: the maximum capacity of the queue
        :param n_actions: the number of actions available in the environment
        :param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param g_value: the type of value to be used, i.e. "reward" or "efe"
        :param discriminator_threshold: the threshold at which an image is considered fake by the discriminator.
        :param steps_done: the number of training iterations performed to date.
        """

        # Initialize the neural networks of the model.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.critic = critic
        self.target = None
        self.synchronize_target()

        # Ensure the neural networks are on the right device.
        self.to_device()

        # Store learning rates.
        self.vfe_lr = vfe_lr
        self.discriminator_lr = discriminator_lr
        self.efe_lr = efe_lr

        # Initialize the optimizers.
        self.vfe_optimizer, self.discriminator_optimizer, self.efe_optimizer = \
            self.get_optimizers(
                self.encoder, self.decoder, self.transition, self.critic,
                self.vfe_lr, self.discriminator_lr, self.efe_lr
            )

        # Beta scheduling.
        self.n_steps_beta_reset = n_steps_beta_reset
        self.beta_starting_step = beta_starting_step
        self.beta = beta
        self.beta_rate = beta_rate

        # Store the other attributes.
        self.total_rewards = 0.0
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.queue_capacity = queue_capacity
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.g_value = g_value
        self.discriminator_threshold = discriminator_threshold
        self.tensorboard_dir = tensorboard_dir

        # Store the number of actions and the one-hot encoding of the actions.
        self.n_actions = n_actions
        self.actions = torch.IntTensor([i for i in range(0, self.n_actions)]).to(Device.get())

        # Create summary writer for monitoring
        self.writer = SummaryWriter(tensorboard_dir)
        self.writer.add_custom_scalars({
            "Actions": {
                "acts_prob": [
                    "Multiline",
                    ["acts_prob/down", "acts_prob/up", "acts_prob/left", "acts_prob/right"]
                ],
                "discriminator_acts": [
                    "Multiline",
                    ["discr_acts/down", "discr_acts/up", "discr_acts/left", "discr_acts/right"]
                ],
            },
        })

    @staticmethod
    def get_optimizers(encoder, decoder, transition, critic, vfe_lr, discriminator_lr, efe_lr):
        """
        Build the optimizers used to train the neural networks of the model.
        :param encoder: the encoder network.
        :param decoder: the decoder network.
        :param transition: the transition network.
        :param critic: the critic network.
        :param vfe_lr: the learning rate of the Variational Free Energy optimizer.
        :param discriminator_lr: the learning rate of the discriminator loss optimizer.
        :param efe_lr: the learning rate of the Expected Free Energy loss optimizer.
        :return: the optimizers of the VFE, discriminator loss and EFE loss.
        """
        # Create the Variational Free Energy optimizer.
        vfe_params = \
            list(encoder.parameters()) + \
            list(decoder.parameters()) + \
            list(transition.parameters())
        vfe_optimizer = Adam(vfe_params, lr=vfe_lr)

        # Create the discriminator loss optimizer.
        discriminator_params = \
            list(encoder.parameters())
        discriminator_optimizer = Adam(discriminator_params, lr=discriminator_lr)

        # Create the Expected Free Energy loss optimizer.
        efe_params = \
            list(critic.parameters())
        efe_optimizer = Adam(efe_params, lr=efe_lr)

        return vfe_optimizer, discriminator_optimizer, efe_optimizer

    def to_device(self):
        """
        Send the models on the right device, i.e. CPU or GPU.
        :return: nothins
        """
        self.encoder.to(Device.get())
        self.decoder.to(Device.get())
        self.transition.to(Device.get())
        self.critic.to(Device.get())
        self.target.to(Device.get())

    def step(self, obs, config):
        """
        Select a random action based on the critic ouput
        :param obs: the input observation from which decision should be made
        :param config: the hydra configuration
        :return: the random action
        """
        # Extract the current state from the current observation.
        obs = torch.unsqueeze(obs, dim=0)
        state, _, _ = self.encoder(obs)

        # Check if some action transition has not been learned yet.
        next_state, _ = self.transition(state.repeat(self.n_actions, 1), self.actions)
        next_obs = self.decode_images(next_state)
        _, _, are_real = self.encoder(next_obs)
        print("Are real? {}".format(are_real))
        are_real = torch.gt(are_real, self.discriminator_threshold)

        # Display whether the agent is exploring or exploiting.
        # Also display the probability that each action leads to a real image
        # according to the discriminator.
        if config["enable_tensorboard"]:
            behavior = 0 if not torch.all(are_real) else 1
            self.writer.add_scalar("Exploration(0) vs Exploitation(1)", behavior, self.steps_done)
            self.writer.add_scalar("discr_acts/down",  are_real[0][0], self.steps_done)
            self.writer.add_scalar("discr_acts/up",    are_real[1][0], self.steps_done)
            self.writer.add_scalar("discr_acts/left",  are_real[2][0], self.steps_done)
            self.writer.add_scalar("discr_acts/right", are_real[3][0], self.steps_done)

        # If some actions do not lead to realistic observations, select one of those actions.
        if not torch.all(are_real):
            print("Exploration!")
            are_real = torch.logical_not(are_real).to(torch.float64)
            are_real = are_real / are_real.sum()
            return Categorical(torch.squeeze(are_real)).sample()

        # Else select the action that maximises the EFE predicted by the critic.
        print("Best action!")
        return self.critic(state).max(1)[1].item()  # Best action.

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

            # Render the environment and monitor total rewards (if needed).
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
        # Synchronize the target with the critic (if needed).
        if self.steps_done % self.n_steps_between_synchro == 0:
            self.synchronize_target()

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

        # Compute the loss of the discriminator.
        discriminator_loss = self.compute_discriminator_loss(config, obs)

        # Perform one step of gradient descent on the discriminator network.
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Implement the cyclical scheduling for beta.
        if self.steps_done >= self.beta_starting_step:
            self.beta = np.clip(self.beta + self.beta_rate, 0, 1)
        if self.steps_done % self.n_steps_beta_reset == 0:
            self.beta = 0

    def compute_discriminator_loss(self, config, obs):
        """
        Compute the loss of the discriminator network.
        :param config: the hydra configuration.
        :param obs: the observation at time step t.
        :return: the loss of the discriminator network.
        """
        # Compute the fake images.
        mean, log_var, are_real_obs = self.encoder(obs)
        fake_obs = self.decode_images(mean).detach()
        _, _, are_real_fake_obs = self.encoder(fake_obs)

        # Compute the loss function of the discriminator.
        loss = nn.BCELoss()
        loss_real = loss(are_real_obs, torch.ones_like(are_real_obs))
        loss_fake = loss(are_real_fake_obs, torch.zeros_like(are_real_fake_obs))
        total_loss = loss_real + loss_fake

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("Discriminator loss", total_loss, self.steps_done)

        return total_loss

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
        mean_hat_t, log_var_hat_t, _ = self.encoder(obs)
        _, log_var = self.transition(mean_hat_t, actions)
        mean_hat, log_var_hat, _ = self.encoder(next_obs)

        # Compute the G-values of each action in the current state.
        critic_pred = self.critic(mean_hat_t)
        critic_pred = critic_pred.gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))

        # For each batch entry where the simulation did not stop,
        # compute the value of the next states.
        future_gval = torch.zeros(config["batch_size"], device=Device.get())
        future_gval[torch.logical_not(done)] = self.target(mean_hat[torch.logical_not(done)]).max(1)[0]
        future_gval = future_gval.detach()

        # Compute the immediate G-value.
        immediate_gval = rewards

        # Add information gain to the immediate g-value (if needed).
        if self.g_value == "efe":
            immediate_gval += mathfc.entropy_gaussian(log_var_hat) - mathfc.entropy_gaussian(log_var)
            immediate_gval = immediate_gval.to(torch.float32)

        # Compute the discounted G values.
        gval = immediate_gval + self.discount_factor * future_gval

        # Compute the loss function.
        loss = nn.SmoothL1Loss()
        efe_loss = loss(critic_pred, gval.unsqueeze(1))

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("EFE loss", efe_loss, self.steps_done)

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
        # TODO optimize the computation of those vector across efe and vfe computation
        mean_hat, log_var_hat, _ = self.encoder(obs)
        states = mathfc.reparameterize(mean_hat, log_var_hat)
        mean_hat, log_var_hat, _ = self.encoder(next_obs)
        next_state = mathfc.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        log_alpha = self.decoder(next_state)

        # Compute the variational free energy.
        kl_div_hs = mathfc.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood = mathfc.log_bernoulli_with_logits(next_obs, log_alpha)
        vfe_loss = self.beta * kl_div_hs - log_likelihood

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("KL_div_hs", kl_div_hs, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood", - log_likelihood, self.steps_done)
            self.writer.add_scalar("Beta", self.beta, self.steps_done)
            self.writer.add_scalar("VFE", vfe_loss, self.steps_done)

        return vfe_loss

    def decode_images(self, states):
        """
        Compute the images corresponding to the input states.
        :param states: the input state that needs to be decoded.
        :return: the decoded images.
        """
        return self.decoder(states).sigmoid()

    def synchronize_target(self):
        """
        Synchronize the target with the critic.
        :return: nothing.
        """
        self.target = copy.deepcopy(self.critic)
        self.target.eval()

    def save(self, config):
        """
        Create a checkpoint file allowing the agent to be reloaded later.
        :param config: the hydra configuration.
        :return: nothing.
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
            "n_actions": config["env"]["n_actions"],
            "decoder_net_state_dict": self.decoder.state_dict(),
            "decoder_net_module": str(self.decoder.__module__),
            "decoder_net_class": str(self.decoder.__class__.__name__),
            "encoder_net_state_dict": self.encoder.state_dict(),
            "encoder_net_module": str(self.encoder.__module__),
            "encoder_net_class": str(self.encoder.__class__.__name__),
            "transition_net_state_dict": self.transition.state_dict(),
            "transition_net_module": str(self.transition.__module__),
            "transition_net_class": str(self.transition.__class__.__name__),
            "critic_net_state_dict": self.critic.state_dict(),
            "critic_net_module": str(self.critic.__module__),
            "critic_net_class": str(self.critic.__class__.__name__),
            "vfe_lr": self.vfe_lr,
            "discriminator_lr": self.discriminator_lr,
            "efe_lr": self.efe_lr,
            "beta": self.beta,
            "n_steps_beta_reset": self.n_steps_beta_reset,
            "beta_starting_step": self.beta_starting_step,
            "beta_rate": self.beta_rate,
            "steps_done": self.steps_done,
            "discount_factor": self.discount_factor,
            "queue_capacity": self.queue_capacity,
            "n_steps_between_synchro": self.n_steps_between_synchro,
            "tensorboard_dir": self.tensorboard_dir,
            "g_value": self.g_value,
            "discriminator_threshold": self.discriminator_threshold,
        }, checkpoint_file)

    @staticmethod
    def load_constructor_parameters(config, checkpoint, training_mode=True):
        """
        Load the constructor parameters from a checkpoint.
        :param config: the hydra configuration.
        :param checkpoint: the chechpoint from which to load the parameters.
        :param training_mode: True if the agent is being loaded for training, False otherwise.
        :return: a dictionary containing the contrutor's parameters.
        """
        return {
            "encoder": Checkpoint.load_encoder(checkpoint, training_mode),
            "decoder": Checkpoint.load_decoder(checkpoint, training_mode),
            "transition": Checkpoint.load_transition(checkpoint, training_mode),
            "critic": Checkpoint.load_critic(checkpoint, training_mode),
            "vfe_lr": checkpoint["vfe_lr"],
            "discriminator_lr": checkpoint["discriminator_lr"],
            "efe_lr": checkpoint["efe_lr"],
            "beta": checkpoint["beta"],
            "n_steps_beta_reset": checkpoint["n_steps_beta_reset"],
            "beta_starting_step": checkpoint["beta_starting_step"],
            "beta_rate": checkpoint["beta_rate"],
            "steps_done": checkpoint["steps_done"],
            "discount_factor": checkpoint["discount_factor"],
            "queue_capacity": checkpoint["queue_capacity"],
            "n_steps_between_synchro": checkpoint["n_steps_between_synchro"],
            "tensorboard_dir": config["agent"]["tensorboard_dir"],
            "g_value": checkpoint["g_value"],
            "discriminator_threshold": checkpoint["discriminator_threshold"],
            "n_actions": checkpoint["n_actions"]
        }
