import numpy as np
from agents.save.Checkpoint import Checkpoint
from environments import EnvFactory
import logging
import hydra
from omegaconf import OmegaConf, open_dict
from environments.wrappers.DefaultWrappers import DefaultWrappers
from representational_similarity.activations import get_activations
from representational_similarity.data import get_batch
from representational_similarity.visualisation import plot_distrib

logger = logging.getLogger("variance_distribution")


def compute_distribs(model, data, save_path):
    df_save_path = "_".join(save_path.split("_")[1:3])
    logger.info("Retrieving layer activations...")
    acts = get_activations(data, model, logvar_only=True)
    acts = {k: np.exp(v) for k, v in acts.items()}
    for l, act in acts.items():
        logger.debug("Activation shape of {}: {}".format(l, act.shape))
        plot_distrib(act, data[1], l, df_save_path)


@hydra.main(config_path="config", config_name="distrib")
def compute_distrib(cfg):
    # Display the configuration.
    logger.info("Experiment config:\n{}".format(OmegaConf.to_yaml(cfg)))
    save_path = "log_var_distrib_{}_{}".format(cfg.agent_name, cfg.agent_seed)

    # Create the environment.
    env = EnvFactory.make(cfg)
    with open_dict(cfg):
        cfg.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, cfg.images.shape)

    # Sample a batch of experiences.
    samples, actions, rewards, done, next_obs = get_batch(batch_size=5000, env=env)

    model = Checkpoint(cfg.agent_tensorboard_dir, cfg.agent_path).load_model()
    compute_distribs(model, (samples, actions), save_path)


if __name__ == "__main__":
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Compute the correlations between the variance of two agents.
    compute_distrib()
