from torch.nn.functional import one_hot
from torch.distributions import MultivariateNormal
from torch import diag, tensor, cat
from algorithms.data.TreeNode import TreeNode as Node


#
# Class implementing the Monte Carlo Tree Search algorithm.
#
class MCTS:

    def __init__(self, nb_actions, nb_planning_steps=100, exploration_constant=5):
        """
        Constructor.
        :param nb_actions: the number of actions in the environment.
        :param nb_planning_steps: the number of planning iterations in each action perception cycle.
        :param exploration_constant:
        """
        self.__nb_actions = nb_actions
        self.__nb_planning_steps = nb_planning_steps
        self.__exploration_constant = exploration_constant
        self.__root = None

    def reset(self, root):
        """
        Reset the tree to its original state.
        :param root: the root of the tree.
        :return: nothing.
        """
        self.__root = root

    # TODO policy_net should be used with improved UCT
    def select_node(self, policy_net):
        """
        Select the node that must be expanded.
        :param policy_net: the policy network mapping state to action.
        :return: the node to expand.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling select_node.")

        # Select the child root with the highest UCT value.
        current = self.__root
        while len(current.children) != 0:
            current = max(current.children, key=lambda x: x.uct(self.__exploration_constant))
        return current

    def expand(self, node, transition_net):
        """
        Expand the children of the input node.
        :param node: the node whose children must be expanded.
        :param transition_net: the mapping from a pair of state and action to the distribution over next the state.
        :return: the expanded nodes.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling expand.")

        # Sample a state from the node's beliefs.
        state = node.beliefs.sample()

        # Expand all children of the node.
        for i in range(self.__nb_actions):
            action = one_hot(tensor(i), self.__nb_actions)
            mean, sigma = transition_net(cat((state, action)))
            node.add_child(Node(MultivariateNormal(mean, diag(sigma)), action=i))
        return node.children

    # TODO policy_net should to perform rollout
    def evaluate_children(self, node, critic_net, policy_net):
        """
        Evaluate the cost of the children.
        :param node: the node whose children must be evaluated.
        :param critic_net: the critic network mapping state to cost of actions.
        :param policy_net: the policy network mapping state to action.
        :return: nothing.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling evaluate_children.")

        # Sample the beliefs
        state = node.beliefs.sample()
        costs = critic_net(state)
        for child in node.children:
            child.cost = costs[child.action]

    def back_propagate(self, nodes):
        """
        Back propagate the minimal cost upward in the tree and increase the number of visits by one all ancestors.
        :param nodes: the newly expanded nodes.
        :return: nothing.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling back_propagate.")

        # Back propagate the cost of the node
        node = min(nodes, key=lambda x: x.cost)
        current = node.parent
        while current is not None:
            current.cost += node.cost
            current.visits += 1
            current = current.parent

    def select_action(self, mode="avg_cost"):
        """
        Select an action according to the specified mode.
        :param mode: mode of action selection. Supported value: uct, avg_cost, visits and cost.
        :return: the selected action.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling select_action.")

        # Select the best action according to UCT value
        if mode == "uct":
            node = max(self.__root.children, key=lambda x: x.uct(self.__exploration_constant))
            return node.action

        # Select the best action according to number of visits
        if mode == "visits":
            node = max(self.__root.children, key=lambda x: x.visits)
            return node.action

        # Select the best action according to cost
        if mode == "cost":
            node = min(self.__root.children, key=lambda x: x.cost)
            return node.action

        # Select the best action according to average cost
        if mode == "avg_cost":
            node = min(self.__root.children, key=lambda x: x.cost / x.visits)
            return node.action

        # Unsupported mode
        raise Exception("Unsupported mode of action selection.")

    @property
    def nb_planning_steps(self):
        """
        Getter.
        :return: the number of planning iterations.
        """
        return self.__nb_planning_steps

    @nb_planning_steps.setter
    def nb_planning_steps(self, n_planning_steps):
        """
        Setter.
        :param n_planning_steps: the new number of planning iterations.
        :return: nothing.
        """
        self.__nb_planning_steps = n_planning_steps
