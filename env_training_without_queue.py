from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from datetime import datetime
from singletons.Logger import Logger
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import numpy as np
import random
import torch
from agents.save.Checkpoint import Checkpoint


@hydra.main(config_path="config", config_name="training")
def train(config):
    # Set the seed requested by the user.
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    Logger.get(name="Training").info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    env = EnvFactory.make(config)
    with open_dict(config):
        config.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, config["images"]["shape"])

    # Create the agent and train it.
    archive = Checkpoint(config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])

    # Retrieve the initial observation from the environment.
    obs = env.reset()

    # Render the environment (if needed).
    if config["debug_mode"]:
        env.render()

    # Train the agent.
    Logger.get().info("Start the training at {time}".format(time=datetime.now()))
    while agent.steps_done < config["n_training_steps"]:

        # Collect a batch of data point.
        batch = []
        for i in range(0, config["batch_size"]):
            # Select an action.
            action = agent.step(obs, config)

            # Execute the action in the environment.
            obs, reward, done, _ = env.step(action)
            batch.append(torch.unsqueeze(obs, dim=0))

            # Reset the environment when a trial ends.
            if done:
                obs = env.reset()

            # Render the environment.
            if config["debug_mode"]:
                env.render()
        batch = torch.cat(batch)

        # Compute the variational free energy.
        vfe_loss = agent.compute_vfe(config, batch)

        # Perform one step of gradient descent on the other networks.
        agent.optimizer.zero_grad()
        vfe_loss.backward()
        agent.optimizer.step()

        # Save the agent (if needed).
        if agent.steps_done % config["checkpoint"]["frequency"] == 0:
            agent.save(config["checkpoint"]["file"])

        # Increase the number of steps done.
        agent.steps_done += 1

    # Close the environment.
    env.close()


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    train()
