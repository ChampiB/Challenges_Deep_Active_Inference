from torch import nn, zeros, eye
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
from networks.encoders.ConvEncoder import ConvEncoder as DefaultEncoder
from networks.decoders.UpConvDecoder import UpConvDecoder as DefaultDecoder
from torch.optim import Adam


#
# Implement a Variational Auto-Encoder.
#
class VAE(nn.Module):

    def __init__(self, n_states, image_shape=(1, 28, 28)):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create default encoder and decoder networks
        self.__encoder = DefaultEncoder(n_states, image_shape)
        self.__decoder = DefaultDecoder(n_states, image_shape, self.__encoder.conv_output_shape())

        # Create optimizer
        self.__optimizer = Adam(self.parameters(), lr=0.0001)

        # Create a multivariate Gaussian with mean zero and variance one
        self.__standard_Gaussian = MultivariateNormal(zeros(n_states), eye(n_states))

        # Check if the GPU is available and move both the encoder and the decoder to the selected device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__encoder.to(device)
        self.__decoder.to(device)

    def sample_z(self, mean, log_variance, n_samples):
        """
        Implement the re-parameterization trick.
        :param mean: the mean of the Gaussian to sample from.
        :param log_variance: the logarithm of the variance of the Gaussian to sample from.
        :param n_samples: the number of samples to generate.
        :return: the sampled hidden representations.
        """
        epsilon = self.__standard_Gaussian.sample(sample_shape=[n_samples])
        return epsilon * log_variance.exp() + mean

    def forward(self, x):
        """
        Implement the forward pass of the VAE, using the re-parameterization.
        :param x: the input.
        :return: the reconstructed input.
        """
        mean, log_variance = self.__encoder(x)
        z = self.sample_z(mean, log_variance, x.shape[0])
        return self.__decoder(z), z, mean, log_variance

    @staticmethod
    def log_normal_pdf(x, mean, log_variance):
        """
        Compute the logarithm of the probability density function (pdf) of a Gaussian distribution.
        :param x: the input.
        :param mean: the mean of the normal distribution.
        :param log_variance: the logarithm of the variance of the normal distribution.
        :return: the logarithm of the pdf of a Gaussian distribution
        """
        dims = [i for i in range(1, len(x.shape))]
        diff = (x - mean)
        return - 0.5 * (np.log(2 * np.pi) + diff * diff * (-log_variance).exp() + log_variance).sum(dim=dims)

    @staticmethod
    def loss(x, reconstruction, z, mean_q, log_variance_q, mean_p, log_variance_p, loss=nn.MSELoss()):
        """
        Compute the Variation Free Energy (VFE).
        :param x: the input.
        :param reconstruction: the reconstructed input.
        :param z: the latent representation of the input.
        :param mean_q: the mean of the posterior.
        :param log_variance_q: the logarithm of the variance of the posterior.
        :param mean_p: the mean of the prior.
        :param log_variance_p: the logarithm of the variance of the prior.
        :param loss: the reconstruction loss to use, i.e. MSELoss() or BCELoss().
        :return: the VFE
        """
        return VAE.beta_loss(x, reconstruction, z, mean_q, log_variance_q, mean_p, log_variance_p, 1, loss)

    @staticmethod
    def beta_loss(x, reconstruction, z, mean_q, log_variance_q, mean_p, log_variance_p, beta, loss=nn.MSELoss()):
        """
        Compute the Variation Free Energy (VFE).
        :param x: the input.
        :param reconstruction: the reconstructed input.
        :param z: the latent representation of the input.
        :param mean_q: the mean of the posterior.
        :param log_variance_q: the logarithm of the variance of the posterior.
        :param mean_p: the mean of the prior.
        :param log_variance_p: the logarithm of the variance of the prior.
        :param beta: the beta parameter of the BetaVAE literature.
        :param loss: the reconstruction loss to use, i.e. MSELoss() or BCELoss().
        :return: the VFE
        """
        log_pz = VAE.log_normal_pdf(z, mean_p, log_variance_p)
        log_qzx = VAE.log_normal_pdf(z, mean_q, log_variance_q)
        log_pxz = - loss(reconstruction, x)
        return - (log_pxz + beta * (log_pz - log_qzx)).mean()

    def training_step(self, x, mean_p, log_variance_p, beta=0.0001):
        """
        Perform one iteration of training
        :param x: the input to train on.
        :param mean_p: the mean of the prior over hidden state.
        :param log_variance_p: the logarithm of the variance over hidden states.
        :param beta: the beta parameters of the beta-VAE.
        :return: a 4-tuples (loss, images, means, log_variances)
        loss: the loss of the training samples
        images: the generated images
        means: the means vector of the posterior for the given samples
        log_variances: the logarithms of the variances of the posterior over hidden states
        """

        # Forward pass
        images, z, means, log_variances = self(x)

        # Create the loss function
        loss = VAE.beta_loss(x, images, z, means, log_variances, mean_p, log_variance_p, beta)

        # Perform one step of gradient descent
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        return loss, images, means, log_variances

    @property
    def optimizer(self):
        """
        Getter.
        :return: the optimiser.
        """
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, opt):
        """
        Setter.
        :param opt: the new optimiser.
        :return: nothing.
        """
        self.__optimizer = opt

    @property
    def encoder(self):
        """
        Getter.
        :return: the encoder network.
        """
        return self.__encoder

    @encoder.setter
    def encoder(self, net):
        """
        Setter.
        :param net: the new encoder network.
        :return: nothing.
        """
        self.__encoder = net

    @property
    def decoder(self):
        """
        Getter.
        :return: the decoder network.
        """
        return self.__decoder

    @decoder.setter
    def decoder(self, net):
        """
        Setter.
        :param net: the decoder network.
        :return: nothing.
        """
        self.__decoder = net
