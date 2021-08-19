#
# Class implementing an Active Inference agent with Actor Critic architecture
# and Monte Carlo Tree Search for planning.
#
class ACAI:

    def __init__(self, mcts, env):
        self.__mcts = mcts
        env.
        self.__policy_net = env.#TODO default
        self.__likelihood_net = #TODO default
        self.__reward_net = #TODO default
        self.__transition_net = #TODO default
        self.__critic_net = #TODO default
        self.__inference_net = #TODO default

    @property
    def inference_net(self):
        return self.__inference_net

    @inference_net.setter
    def inference_net(self, net):
        self.__inference_net = net

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
    def likelihood_net(self):
        return self.__likelihood_net

    @likelihood_net.setter
    def likelihood_net(self, net):
        self.__likelihood_net = net

    @property
    def policy_network(self):
        return self.__policy_net

    @policy_network.setter
    def policy_network(self, net):
        self.__policy_net = net
