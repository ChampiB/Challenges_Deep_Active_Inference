import collections
import numpy as np
from torch import from_numpy

#
# Class storing an experience.
#
Experience = collections.namedtuple('Experience', field_names=['obs', 'action', 'reward', 'done', 'next_obs'])


#
# Class implementing the experience replay buffer.
#
class ExperienceBuffer:

    def __init__(self, capacity=10000, device="cpu"):
        """
        Constructor.
        :param capacity: the number of experience the buffer can store.
        :param device: the device on which the data should be stored when sampled from the buffer.
        """

        self.__buffer = collections.deque(maxlen=capacity)
        self.__device = device

    def __len__(self):
        """
        Getter.
        :return: the number of elements contained in the replay buffer.
        """
        return len(self.__buffer)

    def append(self, experience):
        """
        Add a new experience to the buffer.
        :param experience: the experience to add.
        :return: nothing.
        """
        self.__buffer.append(experience)

    def to_pytorch(self, list_of_arrays):
        """
        Transform a list of numpy array to a pytorch tensor.
        :param list_of_arrays: the input arrays to convert into a pytorch tensor.
        :return: the create tensor.
        """
        array = np.concatenate([np.expand_dims(a, 0) for a in list_of_arrays])
        array = from_numpy(array).to(self.__device)
        return array

    def sample(self, batch_size):
        """
        Sample a batch from the replay buffer.
        :param batch_size: the size of the batch to sample.
        :return: observations, actions, rewards, done, next_observations
        where:
        - observations: the batch of observations.
        - actions: the actions performed.
        - rewards: the rewards received.
        - done: whether the environment stop after performing the actions.
        - next_observations: the observations received after performing the actions.
        """
        indices = np.random.choice(len(self.__buffer), batch_size, replace=False)
        obs, actions, rewards, done, next_obs = zip(*[self.__buffer[idx] for idx in indices])
        return self.to_pytorch(obs), self.to_pytorch(actions), self.to_pytorch(rewards), \
            self.to_pytorch(done), self.to_pytorch(next_obs)
