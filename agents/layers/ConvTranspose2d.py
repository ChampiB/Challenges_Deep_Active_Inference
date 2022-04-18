import torch.nn as nn
import math


#
# Class wrapping the nn.ConvTranspose2d layer to add the padding='same' feature.
#
class ConvTranspose2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1,
        padding_mode='zeros', device=None, dtype=None
    ):
        """
        Constructor of the convolutional transpose layer.
        :param in_channels: see documentation of torch.nn.ConvTranspose2d.
        :param out_channels: see documentation of torch.nn.ConvTranspose2d.
        :param kernel_size: see documentation of torch.nn.ConvTranspose2d.
        :param stride: see documentation of torch.nn.ConvTranspose2d.
        :param padding: in addition to what is described in the documentation of
        torch.nn.ConvTranspose2d, this parameter can be a string equals to either
        'valid' or 'same'. If set to 'valid', then no padding is performed. If set
        to 'same', the layer will ensure that: output shape = input shape * stride.
        :param output_padding: see documentation of torch.nn.ConvTranspose2d.
        :param groups: see documentation of torch.nn.ConvTranspose2d.
        :param bias: see documentation of torch.nn.ConvTranspose2d.
        :param dilation: see documentation of torch.nn.ConvTranspose2d.
        :param padding_mode: see documentation of torch.nn.ConvTranspose2d.
        :param device: see documentation of torch.nn.ConvTranspose2d.
        :param dtype: see documentation of torch.nn.ConvTranspose2d.
        """
        super().__init__()

        # Check that the padding parameter is set to a proper value.
        if isinstance(padding, str) and padding not in ['valid', 'same']:
            raise Exception("In constructor of custom layer: 'ConvTranspose2d', par")

        # If padding is a string, apply zero padding, otherwise apply whatever
        # was requested by the user.
        self.padding = (0, 0) if isinstance(padding, str) else padding

        # Remember if the user asked for a padding same.
        self.padding_same = False
        if isinstance(padding, str) and padding == 'same':
            self.padding_same = True

        # Ensure that stride and kernel_size are tuples.
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Compute the padding to apply if padding same is requested.
        self.p_left = math.floor((kernel_size[0] - stride[0]) / 2)
        self.p_right = kernel_size[0] - stride[0] - self.p_left
        self.p_top = math.floor((kernel_size[1] - stride[1]) / 2)
        self.p_bottom = kernel_size[1] - stride[1] - self.p_left

        # Create the pytorch ConvTranspose2d used to perform the deconvolution.
        self.deconv_layer = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, self.padding, output_padding,
            groups, bias, dilation, padding_mode, device, dtype
        )

    def forward(self, x):
        """
        Compute the forward pass of the custom ConvTranspose2d.
        :param x: the input of the layer.
        :return: the output of the layer.
        """
        out = self.deconv_layer(x)
        return self.apply_padding_same(out) if self.padding_same else out

    def apply_padding_same(self, tensor):
        """
        Apply the padding same to the input tensor.
        :param tensor: the input tensor.
        :return: the output tensor after applying the padding same.
        """
        return tensor[
               :, :,  # Select all batch elements and all channels.
               self.p_left:tensor.shape[2] - self.p_right,
               self.p_top:tensor.shape[3] - self.p_bottom
        ]
