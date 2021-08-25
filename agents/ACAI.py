from torch import nn, eye, zeros
from torch.distributions.multivariate_normal import MultivariateNormal
from networks.DiagonalGaussian import DiagonalGaussian as Gaussian
from algorithms.data.TreeNode import TreeNode as Node
import gym


#
# Class implementing an Active Inference agent with Actor Critic architecture
# and Monte Carlo Tree Search for planning.
#
class ACAI:

    def __init__(self, mcts, vae, env, n_states):
        """
        Constructor.
        :param mcts: the Monte Carlo algorithm to use.
        :param vae: the Variational Auto-Encoder to use.
        :param env: the environment in which the agent will be acting.
        :param n_states: the number of components of the Gaussian over latent variables.
        """

        # Sanity checks
        isinstance(env.action_space, gym.spaces.Discrete)  # only support discrete action space

        # Initialise the prior over initial hidden states
        self.__prior_over_states = MultivariateNormal(zeros(n_states), eye(n_states))

        # Store MCTS algorithm
        self.__mcts = mcts

        # Store VAE
        self.__vae = vae

        # Create default networks
        self.__policy_net = nn.Sequential(
            nn.Linear(n_states, env.action_space.n),
            nn.Softmax()
        )
        self.__reward_net = nn.Sequential(
            nn.Linear(n_states, 1),
            nn.Sigmoid()
        )
        self.__transition_net = nn.Sequential(
            nn.Linear(n_states + env.action_space.n, 10),
            nn.ReLU(),
            Gaussian(10, n_states)
        )
        self.__critic_net = nn.Sequential(
            nn.Linear(n_states, env.action_space.n),
            nn.ReLU()
        )

    # TODO implement an empirical prior over hidden states
    def reset(self, observation):
        """
        Reset the agent to its initial state. This method is called at the beginning of each episode.
        :param observation: the initial observation.
        :return: nothing.
        """

        # Compute the posterior over initial hidden states
        posterior = self.__vae.inference(observation, self.__prior_over_states)

        # Reset MCTS algorithm
        self.__mcts.reset(Node(posterior))

    def step(self, env):
        """
        Perform one action perception cycle in the environment.
        :param env: the environment in which the action should be performed.
        :return: true if the environment is done and false otherwise.
        """

        # Perform planning
        for i in range(self.__mcts.nb_planning_steps):
            node = self.__mcts.select_node(self.__policy_net)
            nodes = self.__mcts.expand(node, self.__transition_net)
            self.__mcts.evaluate_children(node, self.__critic_net, self.__policy_net)
            self.__mcts.back_propagate(nodes)

        # Perform an action in the environment
        action = self.__mcts.select_action()
        observation, reward, done, _ = env.step(action)

        # Perform inference
        self.__prior_over_states = self.__vae.inference(observation, self.__prior_over_states)

        # Store new experience in replay buffer
        # TODO implement it as a FIFO queue

        # Perform learning
        # TODO

        return done

    @property
    def critic_net(self):
        return self.__critic_net

    @critic_net.setter
    def critic_net(self, net):
        self.__critic_net = net

    @property
    def transition_net(self):
        return self.__transition_net

    @transition_net.setter
    def transition_net(self, net):
        self.__transition_net = net

    @property
    def reward_net(self):
        return self.__reward_net

    @reward_net.setter
    def reward_net(self, net):
        self.__reward_net = net

    @property
    def policy_network(self):
        return self.__policy_net

    @policy_network.setter
    def policy_network(self, net):
        self.__policy_net = net
