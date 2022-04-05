import agents.dgn_networks.CriticNetworks
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import importlib


@hydra.main(config_path="config", config_name="training")
def save_and_load_models(config):
    layer = agents.dgn_networks.CriticNetworks.LinearRelu(10, 4)
    class_name = str(layer.__module__) + "." + str(layer.__class__.__name__)
    print(class_name)

    module = importlib.import_module(layer.__module__)
    class_ = getattr(module, layer.__class__.__name__)
    layer = class_(10, 4)
    print(type(layer))

    # Create the agent.
    # agent = instantiate(config["agent"])
    # agent.load(config["checkpoint"]["directory"])

    # Save the agent.
    # TODO agent.save(config["checkpoint"]["save_directory"])


if __name__ == '__main__':
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Save and load various models.
    save_and_load_models()
