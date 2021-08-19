import gym
from algorithms.MCTS import MCTS
from agents.ACAI import ACAI


def train_on_gym_environment(env_name="CartPole-v1"):
    env = gym.make(env_name)
    mcts = MCTS()
    agent = ACAI(mcts, env)
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = agent.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    train_on_gym_environment("CartPole-v1")
