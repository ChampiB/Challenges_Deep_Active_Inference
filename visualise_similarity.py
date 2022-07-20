from agents.save.Checkpoint import Checkpoint
from environments import EnvFactory
import logging
import hydra
from omegaconf import OmegaConf
import pandas as pd
from representational_similarity.visualisation import plot_cka

logger = logging.getLogger("similarity_visualisation")


@hydra.main(config_path="config", config_name="similarity_visualisation")
def visualise_sim(cfg):
    # Display the configuration.
    logger.info("Experiment config:\n{}".format(OmegaConf.to_yaml(cfg)))
    res = pd.read_csv(cfg.input_file, sep="\t", index_col=0)
    plot_cka(res, cfg)


if __name__ == "__main__":
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Visualise the similarity between the layers of two agents.
    visualise_sim()