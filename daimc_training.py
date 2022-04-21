from environments import EnvFactory
from environments.wrappers.DefaultWrappers import DefaultWrappers
from singletons.Device import Device
from singletons.Logger import Logger
from torch.distributions.categorical import Categorical
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import agents.math_fc.functions as mathfc
import numpy as np
import random
import torch
from agents.save.Checkpoint import Checkpoint

np_precision = np.float32


def all_current_frame(envs, agent):
    o0 = torch.cat([
        torch.unsqueeze(agent.encode_reward_to_image(
            torch.from_numpy(env.current_frame()), env.last_r
        ), 0) for env in envs
    ])
    o0 = torch.permute(o0, (0, 3, 1, 2)).to(torch.float32).to(device=Device.get())
    return o0, o0.repeat(4, 1, 1, 1)


def make_batch_dsprites_active_inference(envs, agent):
    # Reset all environments.
    for env in envs:
        env.reset()
        env.last_r = -1.0 + np.random.rand() * 2.0

    # Compute probability and log probability of each action
    o0, o0_repeated = all_current_frame(envs, agent)
    efe = agent.calculate_efe_repeated(o0_repeated)
    p_pi, log_p_pi = agent.softmax_with_log(-efe, 4)

    # Select and take actions in the environment.
    actions = [Categorical(p_pi[i]).sample() for i in range(len(envs))]
    for env, action in zip(envs, actions):
        env.last_r *= 0.95
        env.step(action)
    o1 = all_current_frame(envs, agent)[0]

    return o0, o1, torch.IntTensor(actions).to(device=Device.get()), log_p_pi


@hydra.main(config_path="config", config_name="training")
def train(config):
    # Set the seed requested by the user.
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Create the logger and keep track of the configuration.
    Logger.get(name="Training").info("Configuration:\n{}".format(OmegaConf.to_yaml(config)))

    # Create the environment and apply standard wrappers.
    envs = []
    for i in range(0, 50):
        env = EnvFactory.make(config)
        with open_dict(config):
            config.env.n_actions = env.action_space.n
        env = DefaultWrappers.apply(env, config["images"]["shape"])
        envs.append(env)

    # Create the agent.
    archive = Checkpoint(config, config["checkpoint"]["file"])
    agent = archive.load_model() if archive.exists() else instantiate(config["agent"])

    # Train the agent in the environment.
    for epoch in range(0, 1001):

        if epoch > agent.gamma_delay and agent.gamma < agent.gamma_max:
            agent.gamma += agent.gamma_rate

        for i in range(0, 1000):

            # Create a batch for training the agent
            o0, o1, pi0, log_p_pi = make_batch_dsprites_active_inference(envs, agent)

            # Train critic network
            mean_qs0, logvar_qs0 = agent.encoder(o0)
            qs0 = mathfc.reparameterize(mean_qs0, logvar_qs0)

            kl_pi = agent.compute_critic_loss(qs0, log_p_pi)
            agent.critic_optimizer.zero_grad()
            kl_pi.backward()
            agent.critic_optimizer.step()

            # Compute omega
            omega = agent.compute_omega(kl_pi).reshape(-1, 1)

            # Train transition network
            qs1_mean, qs1_logvar = agent.encoder(o1)
            kl_s, ps1_mean, ps1_logvar = agent.compute_transition_loss(qs0, qs1_mean, qs1_logvar, pi0, omega)

            agent.transition_optimizer.zero_grad()
            kl_s.backward()
            agent.transition_optimizer.step()

            # Train encoder and decoder networks
            vfe = agent.compute_vae_loss(config, o1, ps1_mean, ps1_logvar, omega)

            agent.vae_optimizer.zero_grad()
            vfe.backward()
            agent.vae_optimizer.step()

        if epoch % 2 == 0:
            agent.save(config)


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Train the agent.
    train()
