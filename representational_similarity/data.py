import numpy as np
from agents.memory.ReplayBuffer import ReplayBuffer, Experience


def get_batch(batch_size, env, capacity=50000):
    """
    Collect a batch from the environment.
    :param batch_size: the size of the batch to be generated.
    :param env: the environment from which the samples need to be generated.
    :param capacity: the maximum capacity of the queue.
    :return: the generated batch.
    """

    # Create a replay buffer.
    buffer = ReplayBuffer(capacity=capacity)

    # Generates some experiences.
    for i in range(0, capacity):
        obs = env.reset()
        action = np.random.choice(env.action_space.n)
        next_obs, reward, done, _ = env.step(action)
        buffer.append(Experience(obs, action, reward, done, next_obs))

    # Sample a batch from the replay buffer.
    return buffer.sample(batch_size)
