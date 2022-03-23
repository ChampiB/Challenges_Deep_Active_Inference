from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import numpy as np
import random
import torch
from datetime import datetime
from agents.memory.ReplayBuffer import Experience


@hydra.main(config_path="config", config_name="training")
def train(config):
    # Set the seed requested by the user.
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    Logger.get(name="Training").info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environments and apply standard wrappers.
    envs = []
    for i in range(0, config["nb_envs"]):
        env = EnvFactory.make(config)
        with open_dict(config):
            config.env.n_actions = env.action_space.n
        env = DefaultWrappers.apply(env, config["images"]["shape"])
        envs.append(env)

    # Create the agent.
    agent = instantiate(config["agent"])
    agent.load(config["checkpoint"]["directory"])

    # Retrieve the initial observations from the environments.
    obs = []
    for i in range(0, config["nb_envs"]):
        obs.append(envs[i].reset())

    # Render the environment (if needed).
    if config["debug_mode"]:
        envs[0].render()

    # Train the agent.
    Logger.get().info("Start the training at {time}".format(time=datetime.now()))
    while agent.steps_done < config["n_training_steps"]:

        for i in range(0, config["nb_envs"]):
            # Select an action.
            action = agent.step(obs[i], config)

            # Execute the action in the environment.
            old_obs = obs[i]
            obs[i], reward, done, _ = envs[i].step(action)

            # Add the experience to the replay buffer.
            agent.buffer.append(Experience(old_obs, action, reward, done, obs[i]))

            # Reset the environment when a trial ends.
            if done:
                obs[i] = envs[i].reset()

        # Perform one iteration of training (if needed).
        if len(agent.buffer) >= config["buffer_start_size"]:
            agent.learn(config)

        # Save the agent (if needed).
        if agent.steps_done % config["checkpoint"]["frequency"] == 0:
            agent.save(config["checkpoint"]["directory"])

        # Render the environment.
        if config["debug_mode"]:
            envs[0].render()

        # Increase the number of steps done.
        agent.steps_done += config["nb_envs"]

    # Close the environments.
    for i in range(0, config["nb_envs"]):
        envs[i].close()


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    train()
