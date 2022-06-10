import numpy as np
from representational_similarity import logger


class CKA:
    """ Compute the linear Centered Kernel Alignment (CKA).
    Adapted from the demo code of Kornblith et al. [1]. from
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ_

    :param str name: The name of the metric. Default "CKA".
    :param bool debiased: If True, use the debiased implementation of CKA proposed by Székely et al. [2] instead of the
    standard one. Default False.

    .. [1] Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019, May). Similarity of neural network representations
           revisited. In International Conference on Machine Learning (pp. 3519-3529). PMLR.
    .. [2] Székely, G. J., & Rizzo, M. L. (2014). Partial distance correlation with methods for dissimilarities.
           The Annals of Statistics, 42(6), 2382-2412.
    """
    def __init__(self, name="CKA", debiased=False):
        self._debiased = debiased
        self._name = name

    @property
    def debiased(self):
        return self._debiased

    @debiased.setter
    def debiased(self, debiased):
        self._debiased = debiased

    @property
    def name(self):
        return self._name

    def center(self, x):
        """ Center the matrix `x` so that its mean is 0.
        :param np.array x: The matrix to center
        :return: The centered matrix.
        :rtype: np.array
        """
        if self._debiased:
            x = np.dot(x, x.T)
            n = x.shape[0]
            np.fill_diagonal(x, 0)
            means = np.sum(x, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            x_centred = x - means[None, :]
            x_centred -= means[:, None]
            np.fill_diagonal(x_centred, 0)
        else:
            x_centred = x - np.mean(x, axis=0, keepdims=True)
            x_centred = np.dot(x_centred, np.transpose(x_centred))
        return x_centred

    def cka(self, x, y):
        """ Compute The CKA score between `x` and `y`. Both matrices are assumed to be centered and to contain the same
        number of data examples `n`.

        :param np.array x: A centered matrix of size (n, n)
        :param np.array y: A centered matrix of size (n, n)
        :return: A CKA score between 0 (not similar at all) and 1 (identical).
        :rtype: float
        """
        # Note: this method assumes that kc and lc are the centered kernel values given by cka.center(cka.kernel(.))
        # Compute tr(KcLc) = vec(kc)^T vec(lc), omitting the term (m-1)**2, which is canceled by CKA
        hsic = np.dot(np.ravel(x), np.ravel(y))
        normalization_x = np.linalg.norm(x)
        normalization_y = np.linalg.norm(y)
        cka = hsic / (normalization_x * normalization_y)
        # Divide by the product of the Frobenius norms of kc and lc to get CKA
        logger.debug("CKA = {} / ({} * {}) = {}".format(hsic, normalization_x, normalization_y, cka))
        return cka

    def __call__(self, x, y):
        return self.cka(x, y)
