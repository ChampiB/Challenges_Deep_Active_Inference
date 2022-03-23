import collections
import numpy as np

dSpritesDataset = collections.namedtuple('dSpritesDataset', field_names=['images', 's_sizes', 's_dim', 's_bases'])


#
# Singleton to access the dSprites dataset.
#
class DataSet:

    instance = None

    @staticmethod
    def get(images_archive):
        """
        Getter.
        :param images_archive: the file in which the dataset is stored.
        :return: an object containing the dSprite dataset.
        """
        if DataSet.instance is None:
            dataset = np.load(images_archive, allow_pickle=True, encoding='latin1')
            images = dataset['imgs'].reshape(-1, 64, 64, 1)
            metadata = dataset['metadata'][()]
            s_sizes = metadata['latents_sizes']  # [1 3 6 40 32 32]
            s_dim = s_sizes.size
            s_bases = np.concatenate((metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1, ])))
            s_bases = np.squeeze(s_bases)  # self.s_bases = [737280 245760  40960 1024 32]
            DataSet.instance = dSpritesDataset(images, s_sizes, s_dim, s_bases)
        return DataSet.instance
