from torch import ones, squeeze


#
# Class implementing a node of the tree for the (predictor) MCTS algorithm.
#
class NodePMCTS:

    def __init__(self, state, cost, pi, action=-1):
        """
        Constructor.
        :param state: the most likely states of this node.
        :param cost: the cost of taking each action.
        :param pi: the prior probability of taking each action.
        :param action: the action that led to that node.
        """
        self.__parent = None
        self.__children = []
        self.__visits = ones(pi.numel())
        self.__cost = squeeze(cost)
        self.__pi = squeeze(pi)
        self.__state = squeeze(state)
        self.__action = action
        assert self.__cost.ndim == 1 and self.__pi.ndim == 1 and self.__state.ndim == 1

    def uct(self, exp_const):
        """
        Compute the UCT value of the node.
        :param exp_const: the exploration constant.
        :return: the UCT values of all children.
        """
        return - self.__cost / self.__visits + exp_const * self.__pi / self.__visits

    @property
    def action(self):
        """
        Getter.
        :return: the node's action.
        """
        return self.__action

    @action.setter
    def action(self, value):
        """
        Setter.
        :param value: the new action.
        :return: nothing.
        """
        self.__action = value

    @property
    def children(self):
        """
        Getter.
        :return: the node's children.
        """
        return self.__children

    @children.setter
    def children(self, nodes):
        """
        Setter.
        :param nodes: the new children.
        :return: nothing.
        """
        self.__children = []
        for node in nodes:
            self.add_child(node)

    def add_child(self, child):
        """
        Add a child to the node.
        :param child: the child node to add to the list of children.
        :return: nothing.
        """
        child.parent = self
        self.__children.append(child)

    def nb_children(self):
        """
        Getter.
        :return: the number of children of the node.
        """
        return len(self.__children)

    def remove_child(self, index):
        """
        Remove the children corresponding to the index.
        :param index: the index of the children to be deleted.
        :return: nothing.
        """
        del self.__children[index]

    @property
    def visits(self):
        """
        Getter.
        :return: the number of visits.
        """
        return self.__visits

    @visits.setter
    def visits(self, values):
        """
        Getter.
        :param values: the new number of visits.
        :return: nothing.
        """
        self.__visits = values

    def incr_visits(self, index, value=1):
        """
        Increase the number of visits of the node by the value passed as parameters.
        :param index: the index of the child whose number of visits must be increased.
        :param value: the number of visits to add to the current count.
        :return: nothing.
        """
        self.__visits[index] += value

    @property
    def state(self):
        """
        Getter.
        :return: the state.
        """
        return self.__state

    @state.setter
    def state(self, value):
        """
        Setter.
        :param value: the new most likely state of the node.
        :return: nothing
        """
        self.__state = value

    @property
    def cost(self):
        """
        Getter.
        :return: the cost.
        """
        return self.__cost

    @cost.setter
    def cost(self, value):
        """
        Setter.
        :param value: the new cost value.
        :return: nothing
        """
        self.__cost = value

    @property
    def parent(self):
        """
        Getter.
        :return: the parent node.
        """
        return self.__parent

    @parent.setter
    def parent(self, node):
        """
        Setter.
        :param node: the new parent node.
        :return: nothing.
        """
        self.__parent = node

    @property
    def pi(self):
        """
        Getter.
        :return: the prior probability of selecting each action.
        """
        return self.__pi

    @pi.setter
    def pi(self, values):
        """
        Setter.
        :param values: the new prior probability of selecting each action.
        :return: nothing.
        """
        self.__pi = values
