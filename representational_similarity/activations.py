import numpy as np
from torch import nn
from agents import DQN, CHMM, DAI, HMM
from agents.layers.DiagonalGaussian import DiagonalGaussian
from representational_similarity import logger


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def get_activations(data, model, logvar_only=False):
    """ Load a model and generate a dictionary of the activations obtained from `data`.
    We assume that the activations of each layers are exposed.

    :param np.array data: A (n_examples, n_features) data matrix
    :param model: The model to use
    :param logvar_only: If True, only return the activations of the log variance layers of model
    :return: A tuple containing the loaded model, list of activations, and list of layer names.
    """
    activations = {}
    hooks = []
    layers_info = select_and_get_layers(model, logvar_only)

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


def select_and_get_layers(model, logvar_only=False):
    """ Select the layers of the networks of a given model and retrieve their information.

    :param model: The model to use
    :param logvar_only: If True, only return the activations of the log variance layers of model
    :return: A list of tuple of the form (layer_name, layer)
    """
    layers_info = []
    if not isinstance(model, DQN.DQN):
        curr_layers_info, _ = get_layers(list(model.encoder.modules())[-1], "Encoder", logvar_only=logvar_only)
        layers_info += curr_layers_info
    if isinstance(model, CHMM.CHMM) or isinstance(model, DAI.DAI) or isinstance(model, HMM.HMM):
        curr_layers_info, _ = get_layers(list(model.transition.modules())[1], "Transition", logvar_only=logvar_only)
        layers_info += curr_layers_info
    if (isinstance(model, CHMM.CHMM) or isinstance(model, DAI.DAI)) and not logvar_only:
        curr_layers_info, _ = get_layers(list(model.critic.modules())[1], "Critic")
        layers_info += curr_layers_info
    if (isinstance(model, DQN.DQN) or isinstance(model, DAI.DAI)) and not logvar_only:
        # The policy is not the same for DAI so we change the module index
        # It could be nice to uniformise the initialisation of the models architectures in the future.
        idx = -1 if isinstance(model, DQN.DQN) else 1
        curr_layers_info, _ = get_layers(list(model.policy.modules())[idx], "Policy")
        layers_info += curr_layers_info
    logger.debug("Found layers {}".format(layers_info))
    return layers_info


def get_layers(model, prefix, i=1, logvar_only=False):
    """ Select the layers of a given network and retrieve their information.

    :param model: The model to use
    :param prefix: the name of the network
    :param i: start numbering the layers from i, default 1
    :param logvar_only: If True, only return the activations of the log variance layers of model
    :return: A list of tuple of the form (prefix_i, layer)
    """
    layers_info = []
    if logvar_only is True:
        return [("{}_variance".format(prefix), list(model.modules())[-1])], i
    # Get the layers at the current level and annotate them with a generic name
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, DiagonalGaussian):
            layers_info.append(("{}_{}".format(prefix, i), module))
            i += 1
    return (layers_info, i) if not logvar_only else (layers_info[-1], i)

