import gym
from algorithms.MCTS import MCTS
from algorithms.VAE import VAE
from agents.ACAI import ACAI


def train_on_gym_environment(env_name="CartPole-v1", nb_steps=10, debug=False):
    # Create environment
    env = gym.make(env_name)
    observation = env.reset()

    # Create agent
    n_states = 10
    mcts = MCTS(env.action_space.n, nb_planning_steps=100)
    vae = VAE(n_states)
    agent = ACAI(mcts, vae, env, n_states)
    agent.reset(observation)

    # Action perception cycles
    if debug:
        env.render()
    for _ in range(nb_steps):
        done = agent.step(env)
        if debug:
            env.render()
        if done:
            observation = env.reset()
            agent.reset(observation)
    env.close()


if __name__ == '__main__':
    train_on_gym_environment("CartPole-v1", 10, True)
