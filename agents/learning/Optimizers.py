from torch.optim import Adam


def get_adam(modules, lr):
    """
    Create and returns an adam optimizer.
    :param modules: the modules whose parameters must be optimizers.
    :param lr: the learning rate.
    :return: the adam optimizer.
    """
    params = []
    for module in modules:
        params += list(module.parameters())
    return Adam(params, lr=lr)
