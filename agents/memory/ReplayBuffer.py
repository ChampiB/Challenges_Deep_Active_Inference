import collections
import numpy as np
from torch import cat, FloatTensor, BoolTensor, IntTensor, unsqueeze
from singletons.Device import Device


#
# Class storing an experience.
#
Experience = collections.namedtuple('Experience', field_names=['obs', 'action', 'reward', 'done', 'next_obs'])


#
# Class implementing the experience replay buffer.
#
class ReplayBuffer:

    def __init__(self, capacity=10000):
        """
        Constructor
        :param capacity: the number of experience the buffer can store
        """
        self.__device = Device.get()
        self.__buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        """
        Getter
        :return: the number of elements contained in the replay buffer
        """
        return len(self.__buffer)

    def append(self, experience):
        """
        Add a new experience to the buffer
        :param experience: the experience to add
        :return: nothing
        """
        self.__buffer.append(experience)

    @staticmethod
    def list_to_tensor(tensor_list):
        """
        Transform a list of n dimensional tensors into a tensor with n+1 dimensions
        :param tensor_list: the list of tensors
        :return: the output tensor
        """
        return cat([unsqueeze(tensor, 0) for tensor in tensor_list])

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer
        :param batch_size: the size of the batch to sample
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations
        - actions: the actions performed
        - rewards: the rewards received
        - done: whether the environment stop after performing the actions
        - next_observations: the observations received after performing the actions
        """

        # Sample a batch from the replay buffer.
        indices = np.random.choice(len(self.__buffer), batch_size, replace=False)
        obs, actions, rewards, done, next_obs = zip(*[self.__buffer[idx] for idx in indices])

        # Convert the batch into a torch tensor stored on the proper device.
        return self.list_to_tensor(obs).to(self.__device), \
            IntTensor(actions).to(self.__device), \
            FloatTensor(rewards).to(self.__device), \
            BoolTensor(done).to(self.__device),\
            self.list_to_tensor(next_obs).to(self.__device)
