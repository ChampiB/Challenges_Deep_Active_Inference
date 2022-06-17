from agents.save.Checkpoint import Checkpoint
from torch.utils.tensorboard import SummaryWriter
import copy
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from singletons.Device import Device
from torch import nn, unsqueeze
from agents.learning import Optimizers
import torch


#
# Implement a critic able to evaluate the qualities of each action given
# the true latent space as input.
#
class Critic:

    def __init__(
            self, critic, discount_factor, efe_lr, queue_capacity, action_selection,
            n_steps_between_synchro, tensorboard_dir, steps_done=0, **_
    ):
        """
        Constructor
        :param critic: the critic network
        :param discount_factor: the factor by which the future EFE is discounted.
        :param efe_lr: the learning rate of the critic network
        :param queue_capacity: the maximum capacity of the queue
        :param action_selection: the action selection to be used
        :param n_steps_between_synchro: the number of steps between two synchronisations
            of the target and the critic
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param steps_done: the number of training iterations performed to date.
        """

        # Neural networks.
        self.critic = critic
        self.target = copy.deepcopy(self.critic)
        self.target.eval()

        # Ensure models are on the right device.
        Device.send([self.critic, self.target])

        # Optimizer.
        self.efe_optimizer = Optimizers.get_adam([critic], efe_lr)

        # Miscellaneous.
        self.total_rewards = 0.0
        self.n_steps_between_synchro = n_steps_between_synchro
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.efe_lr = efe_lr
        self.tensorboard_dir = tensorboard_dir
        self.queue_capacity = queue_capacity
        self.n_actions = 4
        self.action_selection = action_selection

        # Create summary writer for monitoring
        self.writer = SummaryWriter(tensorboard_dir)

    def step(self, state, config):
        """
        Select a random action based on the critic ouput
        :param state: the true state of the environment
        :param config: the hydra configuration
        :return: the random action
        """

        # Format the state for the critic.
        state = torch.unsqueeze(state, dim=0)

        # Select an action.
        return self.action_selection.select(self.critic(state), self.steps_done)

    def train(self, env, config):
        """
        Train the agent in the gym environment passed as parameters
        :param env: the environment
        :param config: the hydra configuration
        :return: nothing
        """

        # Retrieve the initial hidden state from the environment.
        env.reset()
        state = env.get_state()

        # Render the environment (if needed).
        if config["display_gui"]:
            env.render()

        # Train the agent.
        Logger.get().info("Start the training at {time}".format(time=datetime.now()))
        while self.steps_done < config["n_training_steps"]:

            # Select an action.
            action = self.step(state, config)

            # Execute the action in the environment.
            old_state = state
            _, reward, done, _ = env.step(action)
            state = env.get_state()

            # Add the experience to the replay buffer.
            self.buffer.append(Experience(old_state, action, reward, done, state))

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
                env.reset()
                state = env.get_state()

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
        state, actions, rewards, done, next_state = self.buffer.sample(config["batch_size"])

        # Compute the expected free energy loss.
        efe_loss = self.compute_critic_loss(config, state, actions, next_state, done, rewards)

        # Perform one step of gradient descent on the critic network.
        self.efe_optimizer.zero_grad()
        efe_loss.backward()
        self.efe_optimizer.step()

    def compute_critic_loss(self, config, state, actions, next_state, done, rewards):
        """
        Compute the expected free energy loss
        :param config: the hydra configuration
        :param state: the states at time t
        :param actions: the actions at time t
        :param next_state: the states at time t + 1
        :param done: did the simulation ended at time t + 1 after performing the actions at time t
        :param rewards: the rewards at time t + 1
        :return: expected free energy loss
        """

        # Compute the G-values of each action in the current state.
        critic_pred = self.critic(state)
        critic_pred = critic_pred.gather(dim=1, index=unsqueeze(actions.to(torch.int64), dim=1))

        # For each batch entry where the simulation did not stop,
        # compute the value of the next states.
        future_gval = torch.zeros(config["batch_size"], device=Device.get())
        future_gval[torch.logical_not(done)] = self.target(next_state[torch.logical_not(done)]).max(1)[0]
        future_gval = future_gval.detach()

        # Compute the immediate G-value.
        immediate_gval = rewards

        # Compute the discounted G values.
        gval = immediate_gval + self.discount_factor * future_gval

        # Compute the loss function.
        loss = nn.SmoothL1Loss()
        return loss(critic_pred, gval.unsqueeze(1))

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
            "n_states": config["agent"]["n_states"],
            "n_actions": config["env"]["n_actions"],
            "critic_net_state_dict": self.critic.state_dict(),
            "critic_net_module": str(self.critic.__module__),
            "critic_net_class": str(self.critic.__class__.__name__),
            "action_selection": dict(self.action_selection),
            "steps_done": self.steps_done,
            "efe_lr": self.efe_lr,
            "discount_factor": self.discount_factor,
            "tensorboard_dir": self.tensorboard_dir,
            "queue_capacity": self.queue_capacity,
            "n_steps_between_synchro": self.n_steps_between_synchro,
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
            "critic": Checkpoint.load_critic(checkpoint, training_mode),
            "efe_lr": checkpoint["efe_lr"],
            "discount_factor": checkpoint["discount_factor"],
            "tensorboard_dir": tb_dir,
            "action_selection": Checkpoint.load_object_from_dictionary(checkpoint, "action_selection"),
            "queue_capacity": checkpoint["queue_capacity"],
            "n_steps_between_synchro": checkpoint["n_steps_between_synchro"],
            "steps_done": checkpoint["steps_done"],
        }

