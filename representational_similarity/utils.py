import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from agents import DQN, CHMM
from agents.DAI import DAI
from agents.layers.DiagonalGaussian import DiagonalGaussian
from representational_similarity import logger


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def get_activations(data, model):
    """ Load a model and generate a dictionary of the activations obtained from `data`.
    We assume that the activations of each layers are exposed.

    :param np.array data: A (n_examples, n_features) data matrix
    :param model: The model to use
    :return: A tuple containing the loaded model, list of activations, and list of layer names.
    """
    activations = {}
    hooks = []
    layers_info = select_and_get_layers(model)

    # Register forward hooks to get the activations of all the layers
    for name, layer in layers_info:
        hooks.append(layer.register_forward_hook(get_activation(name, activations)))

    model.predict(data)
    logger.debug("Activations obtained after prediction: {}".format(activations))

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return activations


def prepare_activations(x):
    """ Flatten the activation values to get a 2D array and values very close to 0 (e.g., 1e-15) to 0.

    :param x: A (n_example, n_features) matrix of activations
    :return: A (n_example, n_features) tensor of activations. If len(n_features) was initially greater than 1,
    n_features = np.prod(n_features).
    """
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
    # Prevent very tiny values from causing underflow in similarity metrics later on
    x[abs(x) < 1.e-7] = 0.
    return np.array(x)


def save_figure(out_fname, dpi=300, tight=True):
    """ Save a matplotlib figure in an `out_fname` file.

    :param str out_fname: Name of the file used to save the figure.
    :param int dpi: Number of dpi, Default 300.
    :param bool tight: If True, use plt.tight_layout() before saving. Default True.
    """
    if tight is True:
        plt.tight_layout()
    plt.savefig(out_fname, dpi=dpi, transparent=True)
    plt.clf()
    plt.cla()
    plt.close()


def select_and_get_layers(model):
    layers_info = []
    if not isinstance(model, DQN.DQN):
        curr_layers_info, _ = get_layers(list(model.encoder.modules())[-1], "Encoder")
        layers_info += curr_layers_info
    if isinstance(model, CHMM.CHMM) or isinstance(model, DAI):
        curr_layers_info, _ = get_layers(list(model.critic.modules())[1], "Critic")
        layers_info += curr_layers_info
    if isinstance(model, DQN.DQN):
        curr_layers_info, _ = get_layers(list(model.policy.modules())[-1], "Policy")
        layers_info += curr_layers_info
    # The policy is not the same for DAI so we change the module index
    # It could be nice to uniformise the initialisation of the models architectures in the future.
    if isinstance(model, DAI):
        curr_layers_info, _ = get_layers(list(model.policy.modules())[1], "Policy")
        layers_info += curr_layers_info
    logger.debug("Found layers {}".format(layers_info))
    return layers_info


def get_layers(model, prefix, i=1):
    layers_info = []
    # Get the layers at the current level and annotate them with a generic name
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, DiagonalGaussian):
            layers_info.append(("{}_{}".format(prefix, i), module))
            i += 1
    return layers_info, i

