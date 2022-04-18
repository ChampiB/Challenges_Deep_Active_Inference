from torch.utils.tensorboard import SummaryWriter
from agents.save.Checkpoint import Checkpoint
import agents.math_fc.functions as math_fc
from datetime import datetime
from singletons.Logger import Logger
from agents.memory.ReplayBuffer import ReplayBuffer, Experience
from singletons.Device import Device
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import agents.math_fc.functions as mathfc
import torch


#
# Implement a deep active inference agents using Monte-Carlo as proposed in [1].
#
# [1] Z. Fountas, N. Sajid, P. A.M. Mediano and K. Friston,
# "Deep active inference agents using Monte-Carlo methods",
# Advances in Neural Information Processing Systems 33 (NeurIPS 2020).
#
class DAIMC:

    def __init__(
            self, encoder, decoder, transition, critic, n_states, n_actions,
            lr_critic, lr_transition, lr_vae,
            a, b, c, d,
            gamma, gamma_rate, gamma_max, gamma_delay,
            beta_s, beta_o,
            efe_deepness, efe_n_samples,
            queue_capacity, tensorboard_dir, steps_done=0, **_
    ):
        """
        Constructor
        :param encoder: the encoder network
        :param decoder: the decoder network
        :param transition: the transition network
        :param critic: the critic network
        :param n_states: the number of latent `states
        :param n_actions: the number of actions available in the environment
        :param lr_critic: the learning rate of the critic network
        :param lr_transition: the learning rate of the transition network
        :param lr_vae: the learning rate of the encoder and decoder networks
        :param a: 'a' parameter used in the computation of omega for top-down attention
        :param b: 'b' parameter used in the computation of omega for top-down attention
        :param c: 'c' parameter used in the computation of omega for top-down attention
        :param d: 'd' parameter used in the computation of omega for top-down attention
        :param gamma: the initial value of the gamma parameter
        :param gamma_rate: the rate at which the gamma parameter increases
        :param gamma_max: the maximum value of the gamma parameter
        :param gamma_delay: the number of epochs before the gamma parameter start increasing
        :param beta_s: the beta parameter for the accuracy terms of the VFE
        :param beta_o: the beta parameter for the complexity terms of the VFE
        :param efe_deepness: the number of steps for which the EFE is computed
        :param efe_n_samples: the number of samples in the Monte-Carlo estimate of the EFE
        :param queue_capacity: the maximum capacity of the queue
        :param tensorboard_dir: the directory in which tensorboard's files will be written
        :param steps_done: the number of training iterations performed to date.
        """

        # Initialize the neural networks of the model.
        self.encoder = encoder
        self.decoder = decoder
        self.transition = transition
        self.critic = critic

        # Ensure the neural networks are on the right device.
        self.to_device()

        # Store learning rates.
        self.lr_critic = lr_critic
        self.lr_transition = lr_transition
        self.lr_vae = lr_vae

        # Store parameters used to compute omega.
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # Store parameters related to the scheduling of the gamma parameter.
        self.gamma = gamma
        self.gamma_rate = gamma_rate
        self.gamma_max = gamma_max
        self.gamma_delay = gamma_delay

        # Beta parameters.
        self.beta_s = beta_s
        self.beta_o = beta_o

        # Initialize the optimizers.
        self.critic_optimizer, self.transition_optimizer, self.vae_optimizer = \
            self.get_optimizers(
                self.encoder, self.decoder, self.transition, self.critic,
                self.lr_critic, self.lr_transition, self.lr_vae
            )

        # Store the number of states and actions and the list of all possible actions.
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions = torch.IntTensor([i for i in range(0, self.n_actions)]).to(Device.get())

        # Store the other attributes.
        self.total_rewards = 0.0
        self.queue_capacity = queue_capacity
        self.buffer = ReplayBuffer(capacity=queue_capacity)
        self.steps_done = steps_done
        self.tensorboard_dir = tensorboard_dir
        self.efe_deepness = efe_deepness
        self.efe_n_samples = efe_n_samples

        # Create summary writer for monitoring
        self.writer = SummaryWriter(tensorboard_dir)

    @staticmethod
    def get_optimizers(encoder, decoder, transition, critic, lr_critic, lr_transition, lr_vae):
        """
        Build the optimizers used to train the neural networks of the model.
        :param encoder: the encoder network.
        :param decoder: the decoder network.
        :param transition: the transition network.
        :param critic: the critic network.
        :param lr_critic: the learning rate of the critic network
        :param lr_transition: the learning rate of the transition network
        :param lr_vae: the learning rate of the encoder and decoder networks
        :return: the optimizers of the VAE, transition and critic.
        """
        # Create the optimizer for the encoder and decoder networks.
        vae_params = \
            list(encoder.parameters()) + \
            list(decoder.parameters())
        vae_optimizer = Adam(vae_params, lr=lr_vae)

        # Create the optimizer for the transition network.
        transition_params = list(transition.parameters())
        transition_optimizer = Adam(transition_params, lr=lr_transition)

        # Create the optimizer for the critic network.
        critic_params = list(critic.parameters())
        critic_optimizer = Adam(critic_params, lr=lr_critic)

        return critic_optimizer, transition_optimizer, vae_optimizer

    def to_device(self):
        """
        Send the models on the right device, i.e. CPU or GPU.
        :return: nothins
        """
        self.encoder.to(Device.get())
        self.decoder.to(Device.get())
        self.transition.to(Device.get())
        self.critic.to(Device.get())

    def step(self, obs, config):
        """
        Select a random action based on the EFE
        :param obs: the input observation from which decision should be made
        :param config: the hydra configuration
        :return: the random action
        """
        # Compute the EFE of each action.
        obs = torch.unsqueeze(obs, dim=0).repeat(self.n_actions, 1, 1, 1)
        efe = self.calculate_efe_repeated(obs)

        # Compute the probability of each action from the EFE.
        prob_action, _ = self.softmax_with_log(-efe, self.n_actions)

        # Sample an action according to action probability.
        return Categorical(prob_action).sample()

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
            obs = self.encode_reward_to_image(obs, reward)

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

    @staticmethod
    def encode_reward_to_image(obs, reward):
        """
        Encode the reward inside the image.
        :param obs: the image, i.e. observation.
        :param reward: the reward.
        :return: the image containing the reward.
        """
        # Adding the reward encoded to the image.
        w = obs.shape[1]
        half_w = int(w / 2)
        if 0.0 <= reward <= 1.0:
            obs[0:3, 0:half_w] = reward
        elif -1.0 <= reward < 0.0:
            obs[0:3, half_w:w] = -reward
        else:
            exit('Reward must be between zero and one but got: ' + str(reward))
        return obs

    def learn(self, config):
        """
        Perform on step of gradient descent on the encoder and the decoder
        :param config: the hydra configuration
        :return: nothing
        """

        # Sample the replay buffer.
        o0, pi0, _, _, o1 = self.buffer.sample(config["batch_size"])

        # Compute the EFE of repeating each action for 'self.deepness' steps.
        o0_repeated = o0.repeat(self.n_actions, 1, 1, 1)
        efe = self.calculate_efe_repeated(o0_repeated)
        _, log_pi = self.softmax_with_log(-efe, self.n_actions)

        # Compute the critic's loss.
        mean, logvar = self.encoder(o0)
        s0 = math_fc.reparameterize(mean, logvar)
        kl_pi = self.compute_critic_loss(s0, log_pi)

        # Perform one step of gradient descent on the critic network.
        self.critic_optimizer.zero_grad()
        kl_pi.backward()
        self.critic_optimizer.step()

        # Compute the omega parameter for top-down attention.
        omega = self.compute_omega(kl_pi).reshape(-1, 1)

        # Train transition network.
        qs1_mean, qs1_logvar = self.encoder(o1)
        kl_s, ps1_mean, ps1_logvar = self.compute_transition_loss(s0, qs1_mean, qs1_logvar, pi0, omega)

        # Perform one step of gradient descent on the transition network.
        self.transition_optimizer.zero_grad()
        kl_s.backward()
        self.transition_optimizer.step()

        # Train encoder and decoder networks.
        vfe = self.compute_vae_loss(config, o1, ps1_mean, ps1_logvar, omega)

        # Perform one step of gradient descent on the encoder and decoder network.
        self.vae_optimizer.zero_grad()
        vfe.backward()
        self.vae_optimizer.step()

    def calculate_efe_repeated(self, o0):
        """
        Calculate the EFE for the four policies of repeating each one of the four
        actions continuously.
        :param o0: the initial observation from which actions start to be taken.
        :return: the EFE of repeating each possible action.
        """
        # Calculate current mean and log variance of the distribution over s_t, and sample
        # a state from this distribution.
        s0, _ = self.encoder(o0)

        # Create one-hot encoding of all available actions.
        n_samples = s0.shape[0]
        a0 = torch.tensor([int(i * self.n_actions / n_samples) for i in range(0, n_samples)])

        # Compute the EFE cumulated after 'steps' number of steps.
        sum_efe = torch.zeros([o0.shape[0]], device=Device.get())
        for t in range(self.efe_deepness):
            efe, s0 = self.calculate_efe(s0, a0)
            sum_efe += efe

        return sum_efe

    def calculate_efe(self, s0, pi0):
        """
        Compute the EFE of performing actions 'pi0' in state 's0'.
        :param s0: the initial state.
        :param pi0: the action taken.
        :return: the EFE.
        """
        # Compute the EFE.
        efe = torch.zeros([s0.shape[0]], device=Device.get())
        ps1_mean = torch.zeros([s0.shape[0]], device=Device.get())
        ps1_logvar = torch.zeros([s0.shape[0]], device=Device.get())

        for _ in range(self.efe_n_samples):
            ps1_mean, ps1_logvar = self.transition(s0, pi0)
            ps1 = mathfc.reparameterize(ps1_mean, ps1_logvar)
            po1 = self.decoder(ps1)
            _, qs1_logvar = self.encoder(po1)

            efe -= self.compute_reward(po1)  # E[log P(o|pi)]
            efe += torch.sum(self.entropy_normal(ps1_logvar), dim=1)  # E[log Q(s|pi)]
            efe -= torch.sum(self.entropy_normal(qs1_logvar), dim=1)  # -E[log Q(s|o,pi)]

        for _ in range(self.efe_n_samples):
            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            mean, log_var = self.transition(s0, pi0)
            po1 = self.decoder(mathfc.reparameterize(mean, log_var))
            efe += torch.sum(self.entropy_bernoulli(po1), dim=[1, 2, 3])

            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            po1 = self.decoder(mathfc.reparameterize(ps1_mean, ps1_logvar))
            efe -= torch.sum(self.entropy_bernoulli(po1), dim=[1, 2, 3])

        return efe / float(self.efe_n_samples), ps1_mean

    @staticmethod
    def entropy_normal(logvar):
        """
        Compute the entropy of a Gaussian distribution.
        :param logvar: the logarithm variance of the distribution.
        :return: the entropy.
        """
        log_2_pi_e = 1.23247435026
        return 0.5 * (log_2_pi_e + logvar)

    @staticmethod
    def entropy_bernoulli(p, displacement=0.00001):
        """
        The entropy of a Bernouilli distribution.
        :param p: the parameters of the distribution.
        :param displacement: small value to avoid taking the logarithm of zero.
        :return: the entropy.
        """
        return - (1 - p) * torch.log(displacement + 1 - p) - p * torch.log(displacement + p)

    @staticmethod
    def log_bernoulli(x, p, displacement=0.00001):
        """
        Compute the log probability of x assuming a Bernoulli distribution with parameter p.
        :param x: the value of the input random variable.
        :param p: the parameters of the Bernoulli distribution.
        :param displacement: small value to avoid taking the logarithm of zero.
        :return: the log probability of x assuming a Bernoulli distribution.
        """
        return x * torch.log(displacement + p) + (1 - x) * torch.log(displacement + 1 - p)

    @staticmethod
    def softmax_with_log(x, n_elem=4, eps=1e-20, temperature=10.0):
        """
        Compute the softmax and log of the softmax of the input vectors.
        :param x: the input vectors.
        :param n_elem: the number of elements in each input vector.
        :param eps: a small value to avoid taking the logarithm of zero.
        :param temperature: the temperature parameter of the softmax function.
        :return: the softmax and log softmax of the input.
        """
        x = x.reshape(-1, n_elem)
        x = x - x.max(dim=1)[0].reshape(-1, 1)  # Normalization
        e_x = torch.exp(x / temperature)
        e_x_sum = e_x.sum(dim=1).reshape(-1, 1)
        return e_x / e_x_sum, x - torch.log(e_x_sum + eps)

    def compute_reward(self, o):
        """
        Compute the reward associated with an observation.
        :param o: the observation whose reward should be evaluated.
        :return: the reward.
        """
        resolution = o.shape[2]
        perfect_reward = torch.zeros((3, resolution, 1), device=Device.get())
        perfect_reward[:, :int(resolution / 2)] = 1.0
        reward = self.log_bernoulli(o[:, 0:3, 0:resolution, :], perfect_reward)
        return torch.mean(reward, dim=[1, 2, 3]) * 10.0

    def compute_omega(self, kl_pi):
        """
        Compute the precision of the transition mapping.
        :param kl_pi: the KL-divergence between the posterior and prior over actions.
        :return:
        """
        return self.a / (1.0 + torch.exp((kl_pi - self.b) / self.c)) + self.d

    def compute_critic_loss(self, s, log_p_pi):
        """
        Compute the loss of the critic.
        :param s: the current state.
        :param log_p_pi: the logarithm of the prior probability over action.
        :return: the critic's loss.
        """
        # Make sure the gradients does not propagate through the input parameters.
        s = s.detach()
        log_p_pi = log_p_pi.detach()

        # Compute the posterior probability of each action
        logit_pi = self.critic(s)
        q_pi = torch.softmax(logit_pi, dim=1)

        # Compute the logarithm of the posterior probability of each action
        log_q_pi = q_pi.log()

        # Compute KL[Q(pi)||P(pi)] where:
        #  - Q(pi) is the posterior over actions
        #  - P(pi) is the prior over actions
        kl_pi = q_pi * (log_q_pi - log_p_pi)
        return kl_pi.sum(dim=1).mean()

    def compute_transition_loss(self, s0, qs1_mean, qs1_logvar, pi0, omega):
        """
        Compute the loss of the transition.
        :param s0: the current state.
        :param qs1_mean: the mean of the posterior over s1
        :param qs1_logvar: the log variance of the posterior over s1
        :param pi0: the actions taken in the environment.
        :param omega: the precision of the transition mapping.
        :return: the loss of the transition network, as well as the mean and log variance output by the transition.
        """
        # Make sure the gradient does not propagate through the input parameters.
        s0 = s0.detach()
        qs1_mean = qs1_mean.detach()
        qs1_logvar = qs1_logvar.detach()
        pi0 = pi0.detach()
        omega = omega.detach()

        # Compute the mean and log variance of P(s1|s0,pi).
        ps1_mean, ps1_logvar = self.transition(s0, pi0)

        # Compute KL[Q(s1)||P(s1|s0,pi)] where:
        #  - Q(s1) is the posterior over the states at time t+1
        #  - P(s1|s0,pi) is the prior over the states at time t+1
        kl_s = mathfc.kl_div_gaussian(
            qs1_mean, qs1_logvar, ps1_mean, ps1_logvar - omega.log()
        )
        return kl_s, ps1_mean, ps1_logvar

    def compute_vae_loss(self, config, o1, ps1_mean, ps1_logvar, omega):
        """
        Compute the loss of the encoder and decoder.
        :param config: the hydra configuration.
        :param o1: the observation at time t+1.
        :param ps1_mean: the mean of the Gaussian over S_{t+1}.
        :param ps1_logvar: the log variance of the Gaussian over S_{t+1}.
        :param omega: the parameter for top-down precision.
        :return: the loss.
        """
        # Make sure the gradient does not propagate through the input parameters.
        ps1_mean = ps1_mean.detach()
        ps1_logvar = ps1_logvar.detach()
        omega = omega.detach()

        # Compute the mean and log variance of Q(s1)
        qs1_mean, qs1_logvar = self.encoder(o1)

        # Sample from Q(s1) and generate the corresponding image.
        qs1 = mathfc.reparameterize(qs1_mean, qs1_logvar)
        po1 = self.decoder(qs1)  # TODO check activation function of decoder

        # Compute E[log P(o1|s1)] where the expectation is with respect to Q(s1).
        logpo1_s1 = mathfc.log_bernoulli_with_logits(o1, po1)

        # Compute kl[Q(s1)||N(s1;0,I)] where:
        #  - Q(s1) is the posterior over the states at time t+1
        #  - N(s1;0,I) is a naive Gaussian prior over the states at time t+1
        kl_s_naive = mathfc.kl_div_gaussian(qs1_mean, qs1_logvar, 0.0, -omega.log())

        # Compute KL[Q(s1)||P(s1|s0,pi)] where:
        #  - Q(s1) is the posterior over the states at time t+1
        #  - P(s1|s0,pi) is the prior over the states at time t+1
        kl_s = mathfc.kl_div_gaussian(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar - omega.log())

        # Compute the variational free energy.
        vfe = - self.beta_o * logpo1_s1 + \
            self.beta_s * (self.gamma * kl_s + (1.0 - self.gamma) * kl_s_naive)

        # Display debug information, if needed.
        if config["enable_tensorboard"] and self.steps_done % 10 == 0:
            self.writer.add_scalar("KL_s", kl_s, self.steps_done)
            self.writer.add_scalar("KL_s_naive", kl_s_naive, self.steps_done)
            self.writer.add_scalar("Gamma", self.gamma, self.steps_done)
            self.writer.add_scalar("Beta_s", self.beta_s, self.steps_done)
            self.writer.add_scalar("Beta_o", self.beta_o, self.steps_done)
            self.writer.add_scalar("Neg_log_likelihood", - logpo1_s1, self.steps_done)
            self.writer.add_scalar("VFE", vfe, self.steps_done)

        return vfe

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
            "lr_critic": self.lr_critic,
            "lr_transition": self.lr_transition,
            "lr_vae": self.lr_vae,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "gamma": self.gamma,
            "gamma_rate": self.gamma_rate,
            "gamma_max": self.gamma_max,
            "gamma_delay": self.gamma_delay,
            "beta_s": self.beta_s,
            "beta_o": self.beta_o,
            "efe_deepness": self.efe_deepness,
            "efe_n_samples": self.efe_n_samples,
            "steps_done": self.steps_done,
            "queue_capacity": self.queue_capacity,
            "tensorboard_dir": self.tensorboard_dir,
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
            "n_states": checkpoint["n_states"],
            "n_actions": checkpoint["n_actions"],
            "lr_critic": checkpoint["lr_critic"],
            "lr_transition": checkpoint["lr_transition"],
            "lr_vae": checkpoint["lr_vae"],
            "a": checkpoint["a"],
            "b": checkpoint["b"],
            "c": checkpoint["c"],
            "d": checkpoint["d"],
            "gamma": checkpoint["gamma"],
            "gamma_rate": checkpoint["gamma_rate"],
            "gamma_max": checkpoint["gamma_max"],
            "gamma_delay": checkpoint["gamma_delay"],
            "beta_s": checkpoint["beta_s"],
            "beta_o": checkpoint["beta_o"],
            "efe_deepness": checkpoint["efe_deepness"],
            "efe_n_samples": checkpoint["efe_n_samples"],
            "queue_capacity": checkpoint["queue_capacity"],
            "tensorboard_dir": config["agent"]["tensorboard_dir"],
            "steps_done": checkpoint["steps_done"]
        }
