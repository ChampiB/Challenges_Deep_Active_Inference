import time
from agents.learning import Optimizers
from agents.save.Checkpoint import Checkpoint
from torch.utils.tensorboard import SummaryWriter
import copy
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
import agents.math_fc.functions as mathfc
from singletons.Device import Device
from torch import nn, unsqueeze
import torch


#
# Implement a Critical HMM with two latent spaces:
# - one for reward/efe prediction;
# - one for modelling the world.
#
class CHMM_2LS:

    def __init__(
            self, encoder, decoder, transition, critic, discount_factor, beta, efe_lr,
            vfe_lr, queue_capacity, n_steps_between_synchro, tensorboard_dir, g_value,
            action_selection, steps_done=0, **_
    ):
        """
        Constructor
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param critic: the critic network
        :param action_selection: the action selection to be used
        :param discount_factor: the factor by which the future EFE is discounted.
        :param beta: the initial value for beta
        :param efe_lr: the learning rate of the critic network
        :param vfe_lr: the learning rate of the other networks
        :param queue_capacity: the maximum capacity of the queue
        :param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param g_value: the type of value to be used, i.e. "reward" or "efe"
        :param steps_done: the number of training iterations performed to date.
        """

        # Neural networks.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.critic = critic
        self.target = copy.deepcopy(self.critic)
        self.target.eval()

        # Ensure models are on the right device.
        Device.send([self.encoder, self.decoder, self.transition, self.critic, self.target])

        # Optimizers.
        self.vfe_optimizer = Optimizers.get_adam([encoder, decoder, transition], vfe_lr)
        self.efe_optimizer = Optimizers.get_adam([encoder, critic], efe_lr)

        # Miscellaneous.
        self.total_rewards = 0.0
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.g_value = g_value
        self.vfe_lr = vfe_lr
        self.efe_lr = efe_lr
        self.tensorboard_dir = tensorboard_dir
        self.queue_capacity = queue_capacity
        self.action_selection = action_selection
        self.beta = beta

        # Create summary writer for monitoring
        self.writer = SummaryWriter(tensorboard_dir)

    def step(self, obs, config):
        """
        Select a random action based on the critic ouput
        :param obs: the input observation from which decision should be made
        :param config: the hydra configuration
        :return: the random action
        """

        # Extract the current state from the current observation.
        obs = torch.unsqueeze(obs, dim=0)
        _, _, state, _ = self.encoder(obs)

        # Select an action.
        return self.action_selection.select(self.critic(state), self.steps_done)

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
        m_mean_hat_t, _, r_mean_hat_t, _ = self.encoder(obs)
        _, m_log_var = self.transition(m_mean_hat_t, actions)
        _, m_log_var_hat, r_mean_hat, _ = self.encoder(next_obs)

        # Compute the G-values of each action in the current state.
        critic_pred = self.critic(r_mean_hat_t)
        critic_pred = critic_pred.gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))

        # For each batch entry where the simulation did not stop,
        # compute the value of the next states.
        future_gval = torch.zeros(config["batch_size"], device=Device.get())
        future_gval[torch.logical_not(done)] = self.target(r_mean_hat[torch.logical_not(done)]).max(1)[0]

        # Compute the immediate G-value.
        immediate_gval = rewards

        # Add information gain to the immediate g-value (if needed).
        if self.g_value == "efe":
            immediate_gval += mathfc.entropy_gaussian(m_log_var_hat) - mathfc.entropy_gaussian(m_log_var)
            immediate_gval = immediate_gval.to(torch.float32)

        # Compute the discounted G values.
        gval = immediate_gval + self.discount_factor * future_gval
        gval = gval.detach()

        # Compute the loss function.
        loss = nn.SmoothL1Loss()
        return loss(critic_pred, gval.unsqueeze(1))

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
        mean_hat, log_var_hat, _, _ = self.encoder(obs)
        states = mathfc.reparameterize(mean_hat, log_var_hat)
        mean_hat, log_var_hat, _, _ = self.encoder(next_obs)
        next_state = mathfc.reparameterize(mean_hat, log_var_hat)
        mean, log_var = self.transition(states, actions)
        alpha = self.decoder(next_state)

        # Compute the variational free energy.
        kl_div_hs = mathfc.kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
        log_likelihood = mathfc.log_bernoulli_with_logits(next_obs, alpha)
        vfe_loss = self.beta * kl_div_hs - log_likelihood

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("KL_div_hs", kl_div_hs, self.steps_done)
            self.writer.add_scalar("neg_log_likelihood", - log_likelihood, self.steps_done)
            self.writer.add_scalar("Beta", self.beta, self.steps_done)
            self.writer.add_scalar("VFE", vfe_loss, self.steps_done)

        return vfe_loss

    def synchronize_target(self):
        """
        Synchronize the target with the critic.
        :return: nothing.
        """
        self.target = copy.deepcopy(self.critic)
        self.target.eval()

    def save(self, config):
        """
        Create a checkpoint file allowing the agent to be reloaded later
        :param config: the hydra configuration
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
            "n_states": config["agent"]["n_model_states"],
            "n_reward_states": config["agent"]["n_reward_states"],
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
            "beta": self.beta,
            "steps_done": self.steps_done,
            "g_value": self.g_value,
            "vfe_lr": self.vfe_lr,
            "efe_lr": self.efe_lr,
            "discount_factor": self.discount_factor,
            "tensorboard_dir": self.tensorboard_dir,
            "queue_capacity": self.queue_capacity,
            "n_steps_between_synchro": self.n_steps_between_synchro,
            "action_selection": dict(self.action_selection),
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
            "encoder": Checkpoint.load_encoder_2ls(checkpoint, training_mode),
            "decoder": Checkpoint.load_decoder(checkpoint, training_mode),
            "transition": Checkpoint.load_transition(checkpoint, training_mode),
            "critic": Checkpoint.load_critic(checkpoint, training_mode, n_states_key="n_reward_states"),
            "vfe_lr": checkpoint["vfe_lr"],
            "efe_lr": checkpoint["efe_lr"],
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "beta": checkpoint["beta"],
            "discount_factor": checkpoint["discount_factor"],
            "tensorboard_dir": config["agent"]["tensorboard_dir"],
            "g_value": checkpoint["g_value"],
            "queue_capacity": checkpoint["queue_capacity"],
            "n_steps_between_synchro": checkpoint["n_steps_between_synchro"],
            "steps_done": checkpoint["steps_done"]
        }

    # TODO
    def step_test(self, obs, config):
        """
        Select a random action based on the critic ouput
        :param obs: the input observation from which decision should be made
        :param config: the hydra configuration
        :return: the random action
        """

        # Extract the current state from the current observation.
        obs = torch.unsqueeze(obs, dim=0)
        state, _ = self.encoder(obs)

        time.sleep(0.2)

        return self.critic(state).argmax()

        # TODO # Reset MCTS algorithm
        # TODO pi = softmax(self.critic(state), dim=1)
        # TODO cost = self.critic(state)
        # TODO self.mcts.reset(state, cost, pi)

        # TODO # Planning.
        # TODO for i in range(self.mcts.max_planning_steps):
        # TODO     s_node = self.mcts.select_node()
        # TODO     e_node = self.mcts.expand_and_evaluate(s_node, self.transition, self.critic, self.critic)
        # TODO     self.mcts.back_propagate(e_node)

        # TODO # Select the action to perform in the environment.
        # TODO return self.mcts.select_action()

    # TODO
    def test(self, env, config):
        """
        Test the agent in the gym environment passed as parameters
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
        Logger.get().info("Start the testing at {time}".format(time=datetime.now()))
        self.steps_done = 0
        while self.steps_done < config["n_testing_steps"]:

            # Select an action.
            action = self.step_test(obs, config)

            # Execute the action in the environment.
            obs, reward, done, _ = env.step(action)

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