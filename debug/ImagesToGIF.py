import numpy as np
import imageio
from torchvision import transforms


#
# Class converting images to GIF
#
class ImagesToGIF:

    @staticmethod
    def convert(images, file_name='data/outputs/generated_images.gif'):
        """
        Convert the list of grid of images into a gif displaying the grids one after the other.
        :param images: the list of grid of images.
        :param file_name: the name of the output file.
        :return: nothing.
        """
        to_pil_image = transforms.ToPILImage()
        images = [np.array(to_pil_image(img)) for img in images]
        imageio.mimsave(file_name, images)
