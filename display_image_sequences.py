from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Logger import Logger
import torch
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate


# TODO add this to the GUI
@hydra.main(config_path="config", config_name="training")
def display_images(config):
    # Create the logger and keep track of the configuration.
    Logger.get().info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    Logger.get().info("Load the environment...\n")
    env = EnvFactory.make(config)
    with open_dict(config):
        config.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, config["images"]["shape"])

    # Load the agent from the checkpoint.
    Logger.get().info("Load the agent...\n")
    agent = instantiate(config["agent"])
    agent.load(config["checkpoint"]["file"])

    # Collect the initial image from the environment and infer the associated state.
    obs = env.reset()
    obs = torch.unsqueeze(obs, dim=0)
    mean, log_var = agent.encoder(obs)
    next_state = agent.reparameterize(mean, log_var)

    # Collect images from the environment.
    Logger.get().info("Gather images from the environment...\n")
    images = []
    actions = [torch.tensor([1])] * 5
    for action in actions:

        # Generate an images using the VAE.
        image = agent.decoder(next_state)
        image = torch.nn.Sigmoid()(image)
        images.append(obs)
        images.append(image)

        # Take an action in the environment.
        obs, _, _, _ = env.step(action)
        obs = torch.unsqueeze(obs, dim=0)

        # Make the agent imagine what would append if it was taking the action.
        mean, log_var = agent.transition(next_state, action)
        next_state = agent.reparameterize(mean, log_var)

    # Display the images in tensorboard.
    Logger.get().info("Display images...\n")
    images = torch.cat(images, dim=0)
    agent.writer.add_images("An example of (true and generated) trajectories", images)
    Logger.get().info("End.\n")


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the DGN.
    display_images()
