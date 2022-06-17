from agents.save.Checkpoint import Checkpoint
from environments import EnvFactory
import logging
import hydra
from omegaconf import OmegaConf, open_dict
from environments.wrappers.DefaultWrappers import DefaultWrappers
from representational_similarity.cka import CKA
from representational_similarity.utils import get_activations, prepare_activations, save_figure, select_and_get_layers
import pandas as pd
import numpy as np
import seaborn as sns
from agents.memory.ReplayBuffer import ReplayBuffer, Experience

logger = logging.getLogger("similarity_metric")


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


def plot(res, save_path):
    # When we have FC/conv + activation function, we only keep the activation function.
    # We also drop activations from dropout and reshape layers as they are not very informative.
    logger.debug("Dataframe before pre-processing: {}".format(res))
    cols_to_keep = {"Encoder_2": "Encoder_1", "Encoder_4": "Encoder_2", "Encoder_6": "Encoder_3",
                    "Encoder_8": "Encoder_4", "Encoder_11": "Encoder_5", "Encoder_12": "Encoder_6",
                    "Encoder_13": "Encoder_7", "Critic_2": "Critic_1", "Critic_4": "Critic_2", "Critic_6": "Critic_3",
                    "Critic_7": "Critic_4", "Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                    "Policy_7": "Policy_4"}
    rows_to_keep = cols_to_keep
    if res.index.name == "DQN":
        rows_to_keep = {"Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                        "Policy_8": "Policy_4", "Policy_11": "Policy_5", "Policy_14": "Policy_6"}
    if res.columns.name == "DQN":
        cols_to_keep = {"Policy_2": "Policy_1", "Policy_4": "Policy_2", "Policy_6": "Policy_3",
                        "Policy_8": "Policy_4", "Policy_11": "Policy_5", "Policy_14": "Policy_6"}
    df = res.loc[res.index.isin(rows_to_keep.keys()), res.columns.isin(cols_to_keep.keys())]
    df.rename(columns=cols_to_keep, index=rows_to_keep, inplace=True)
    logger.debug("Dataframe after pre-processing: {}".format(df))
    ax = sns.heatmap(df, vmin=0, vmax=1, annot_kws={"fontsize": 13})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_figure("{}.pdf".format(save_path))


def compute_similarity_metric(model1, model2, samples, save_path, m1_name, m2_name):
    logger.info("Instantiating CKA...")
    metric = CKA()
    acts1 = get_activations(samples, model1)
    acts2 = get_activations(samples, model2)
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
    save_path = "CKA_{}_{}".format(cfg.a1_name, cfg.a2_name)

    # Create the environment.
    env = EnvFactory.make(cfg)
    with open_dict(cfg):
        cfg.env.n_actions = env.action_space.n
    env = DefaultWrappers.apply(env, cfg.images.shape)

    # Sample a batch of experiences.
    samples, actions, rewards, done, next_obs = get_batch(batch_size=5000, env=env)

    m1 = Checkpoint(cfg.a1_tensorboard_dir, cfg.a1_path).load_model()
    m2 = Checkpoint(cfg.a2_tensorboard_dir, cfg.a2_path).load_model()
    compute_similarity_metric(m1, m2, samples, save_path, cfg.a1_name, cfg.a2_name)


if __name__ == "__main__":
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Compute the similarity between the layers of two agents.
    compute_sim()
