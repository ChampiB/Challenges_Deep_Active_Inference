import math
import weakref


#
# Class implementing a node of the tree for the MCTS algorithm.
#
class TreeNode:

    def __init__(self, beliefs, action=-1):
        """
        Constructor.
        """
        self.__parent = None
        self.__children = []
        self.__visits = 1
        self.__cost = 0
        self.__beliefs = beliefs
        self.__action = action

    def uct(self, exp_const):
        """
        Compute the UCT value of the node.
        :param exp_const: the exploration constant.
        :return: the UCT value.
        """
        return - self.__cost / self.__visits + exp_const * math.sqrt(math.log(self.__parent.visits) / self.__visits)

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
    def visits(self, value):
        """
        Getter.
        :param value: the new number of visits.
        :return: nothing.
        """
        self.__visits = value

    def incr_visits(self, value=1):
        """
        Increase the number of visits of the node by the value passed as parameters.
        :param value: the number of visits to add to the current count.
        :return: nothing.
        """
        self.__visits += value

    @property
    def beliefs(self):
        """
        Getter.
        :return: the beliefs.
        """
        return self.__beliefs

    @beliefs.setter
    def beliefs(self, distribution):
        """
        Setter.
        :param distribution: the new distribution encoding the beliefs.
        :return: nothing
        """
        self.__beliefs = distribution

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
