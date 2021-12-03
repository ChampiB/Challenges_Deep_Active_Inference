import torch
from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax
from agents.planning.NodePMCTS import NodePMCTS as Node


#
# Class implementing the Monte Carlo Tree Search algorithm.
#
class PMCTS:

    def __init__(self, hp):
        """
        Constructor.
        :param hp: the hyper-parameters of the algorithm.
        """
        self.__hp = hp
        self.__root = None

    def reset(self, root):
        """
        Reset the tree to its original state.
        :param root: the root of the tree.
        :return: nothing.
        """
        self.__root = root

    def select_node(self):
        """
        Select the node that must be expanded.
        :return: the node to expand.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling select_node.")

        # Select the child root with the highest UCT value.
        current = self.__root
        while len(current.children) != 0:
            best_action = torch.argmax(current.uct(self.__hp["zeta"]))
            child = next(filter(lambda c: c.action == best_action, current.children), None)
            if child is None:
                return current
            current = child
        return current

    def expand_and_evaluate(self, node, transition_net, critic_net, policy_net):
        """
        Expand the children of the input node.
        :param node: the node whose children must be expanded.
        :param transition_net: the mapping from a pair of state and action to the distribution over next the state.
        :param critic_net: the critic network mapping state to cost of actions.
        :param policy_net: the policy network mapping state to the probability of each action.
        :return: the expanded nodes.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling expand.")

        # Get the index of the child with the largest UCT value
        best_action = torch.argmax(node.uct(self.__hp["zeta"]))

        # Create the new (expanded) node
        new_state, _ = transition_net(node.state, best_action)
        cost = critic_net(new_state)
        pi = policy_net(new_state)
        new_node = Node(new_state, cost, pi, best_action)

        # Add the new node in the tree
        node.add_child(new_node)
        return new_node

    def back_propagate(self, node):
        """
        Back propagate the cost upward in the tree and increase the number of
        visits of all the ancestors by one.
        :param node: the newly expanded node.
        :return: nothing.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling back_propagate.")

        # Set the current node to be the parent of the newly expanded node
        current = node.parent
        if current is None:
            return

        # Get the action that led to the newly expanded node as well as its cost.
        action = node.action
        cost = current.cost[action]

        # Increase the number of visits of the newly expanded node.
        current.visits[action] += 1

        # Back propagate the cost in the tree, and update the number of visits of
        # the ancestors of the newly expanded node.
        action = current.action
        current = node.parent
        while current is not None:
            current.cost[action] += cost
            current.visits[action] += 1
            action = current.action
            current = current.parent

    def prior_over_actions(self):
        """
        Getter returning the parameters of the categorical distribution
        representing the prior over actions.
        :return: the parameters of the prior over actions.
        """
        return softmax(self.__hp["phi"] * self.__root.visits, dim=0)

    def select_action(self):
        """
        Select an action to be performed in the environment.
        :return: the selected action.
        """

        # Check that the root has been initialised
        if self.__root is None:
            raise Exception("The function reset must be called before calling select_action.")

        # Select the an action according to the planning procedure
        return Categorical(self.prior_over_actions()).sample()
