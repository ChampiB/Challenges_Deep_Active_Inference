import numpy as np
from agents.save.Checkpoint import Checkpoint
from environments import EnvFactory
import logging
import hydra
from omegaconf import OmegaConf, open_dict
from environments.wrappers.DefaultWrappers import DefaultWrappers
from representational_similarity.activations import get_activations
from representational_similarity.data import get_batch
from representational_similarity.visualisation import plot_distrib, plot_corr

logger = logging.getLogger("variance_correlation")


def compute_correlations(model1, model2, data, save_path, m1_name, m2_name):
    df1_save_path = "_".join(save_path.split("_")[1:3])
    df2_save_path = "_".join(save_path.split("_")[3:])
    done = False
    logger.info("Retrieving layer activations...")
    acts1 = get_activations(data, model1, logvar_only=True)
    acts2 = get_activations(data, model2, logvar_only=True)
    acts1 = {k: np.exp(v) for k, v in acts1.items()}
    acts2 = {k: np.exp(v) for k, v in acts2.items()}
    for l1, act1 in acts1.items():
        plot_distrib(act1, l1, df1_save_path)
        logger.debug("Activation shape of {}: {}".format(l1, act1.shape))
        for l2, act2 in acts2.items():
            if not done:
                plot_distrib(act2, l2, df2_save_path)
            logger.info("Computing correlation of {} and {}".format(l1, l2))
            logger.debug("Activation shape of {}: {}".format(l2, act2.shape))
            res = np.corrcoef(act1.T, act2.T)[:act1.T.shape[0], -act2.T.shape[0]:]
            plot_corr(res, m1_name, m2_name, "{}_{}_{}".format(save_path, l1, l2))
        done = True


@hydra.main(config_path="config", config_name="correlation")
def compute_corr(cfg):
    # Display the configuration.
    logger.info("Experiment config:\n{}".format(OmegaConf.to_yaml(cfg)))
    save_path = "Correlation_{}_{}_{}_{}".format(cfg.a1_name, cfg.a1_seed, cfg.a2_name, cfg.a2_seed)

    # Create the environment.
    env = EnvFactory.make(cfg)
    with open_dict(cfg):
        cfg.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, cfg.images.shape)

    # Sample a batch of experiences.
    samples, actions, rewards, done, next_obs = get_batch(batch_size=5000, env=env)

    m1 = Checkpoint(cfg.a1_tensorboard_dir, cfg.a1_path).load_model()
    m2 = Checkpoint(cfg.a2_tensorboard_dir, cfg.a2_path).load_model()
    compute_correlations(m1, m2, (samples, actions), save_path, cfg.a1_name, cfg.a2_name)


if __name__ == "__main__":
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Compute the correlations between the variance of two agents.
    compute_corr()
