from agents.save.Checkpoint import Checkpoint
from environments import EnvFactory
import logging
import hydra
from omegaconf import OmegaConf, open_dict
from environments.wrappers.DefaultWrappers import DefaultWrappers
from representational_similarity.cka import CKA
from representational_similarity.activations import get_activations, prepare_activations
import pandas as pd
from representational_similarity.data import get_batch
from representational_similarity.visualisation import plot

logger = logging.getLogger("similarity_metric")


def compute_similarity_metric(model1, model2, data, save_path, m1_name, m2_name):
    logger.info("Instantiating CKA...")
    metric = CKA()
    acts1 = get_activations(data, model1)
    acts2 = get_activations(data, model2)
    logger.info("Preparing layer activations...")
    f = lambda x: metric.center(prepare_activations(x))
    acts1 = {k: f(v) for k, v in acts1.items()}
    acts2 = {k: f(v) for k, v in acts2.items()}
    res = {}
    for l1, act1 in acts1.items():
        res[l1] = {}
        for l2, act2 in acts2.items():
            logger.info("Computing similarity of {} and {}".format(l1, l2))
            res[l1][l2] = float(metric(act1, act2))
    res = pd.DataFrame(res).T
    # Save csv with m1 layers as header, m2 layers as indexes
    res = res.rename_axis(m2_name.upper(), axis="columns")
    res = res.rename_axis(m1_name.upper())
    res.to_csv("{}.tsv".format(save_path), sep="\t")
    plot(res, save_path)


@hydra.main(config_path="config", config_name="similarity")
def compute_sim(cfg):
    # Display the configuration.
    logger.info("Experiment config:\n{}".format(OmegaConf.to_yaml(cfg)))
    save_path = "CKA_{}_{}_{}_{}".format(cfg.a1_name, cfg.a1_seed, cfg.a2_name, cfg.a2_seed)

    # Create the environment.
    env = EnvFactory.make(cfg)
    with open_dict(cfg):
        cfg.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, cfg.images.shape)

    # Sample a batch of experiences.
    samples, actions, rewards, done, next_obs = get_batch(batch_size=5000, env=env)

    m1 = Checkpoint(cfg.a1_tensorboard_dir, cfg.a1_path).load_model()
    m2 = Checkpoint(cfg.a2_tensorboard_dir, cfg.a2_path).load_model()
    compute_similarity_metric(m1, m2, (samples, actions), save_path, cfg.a1_name, cfg.a2_name)


if __name__ == "__main__":
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Compute the similarity between the layers of two agents.
    compute_sim()
