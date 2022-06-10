import logging
import hydra
from omegaconf import OmegaConf
from representational_similarity.cka import CKA
from representational_similarity.utils import get_activations, prepare_activations, save_figure
import pandas as pd
import seaborn as sns
logger = logging.getLogger("similarity_metric")


def compute_similarity_metric(model1, model2, samples, save_path):
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
    res = res.rename_axis("m1", axis="columns")
    res = res.rename_axis("m2")
    sns.heatmap(res, vmin=0, vmax=1, annot_kws={"fontsize": 13})
    save_figure("{}.pdf".format(save_path))
    res.to_csv("{}.tsv".format(save_path), sep="\t")


@hydra.main(config_path="config", config_name="similarity")
def compute_sim(cfg):
    logger.info("Experiment config:\n{}".format(OmegaConf.to_yaml(cfg)))
    save_path = "CKA_{}_{}.tsv".format(cfg.agent1_name, cfg.agent2_name)
    # TODO: Check how to randomly sample cfg.n_samples from a given dataset
    # This would be equivalent to vae_ld's following code
    # dataset = instantiate(cfg.dataset)
    # samples = dataset.sample(cfg.n_samples, random_state, unique=True)[1]
    samples = None
    # TODO: Check how to load the models
    m1 = None
    m2 = None
    compute_similarity_metric(m1, m2, samples, save_path)


if __name__ == "__main__":
    compute_sim()
