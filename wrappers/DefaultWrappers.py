from wrappers.FireResetWrapper import FireResetWrapper
from wrappers.NormalizePixelsWrapper import NormalizePixelsWrapper
from wrappers.ToPytorchFormatWrapper import ToPytorchFormatWrapper
from wrappers.ProcessFrameWrapper import ProcessFrameWrapper
from wrappers.BufferWrapper import BufferWrapper


#
# Class applying all the default wrappers to an environment.
#
class DefaultWrappers:

    @staticmethod
    def apply(env, image_shape):
        """
        Apply all the default wrapper to the environment.
        :param env: the environment to wrap.
        :param image_shape: the shape of the input images.
        :return: the wrapped environment.
        """

        try:
            env = FireResetWrapper(env)
        except RuntimeError:
            print("FireResetWrapper was not applied to the environment.")
        env = ProcessFrameWrapper(env, image_shape)
        env = ToPytorchFormatWrapper(env)
        env = BufferWrapper(env, image_shape[0])
        env = NormalizePixelsWrapper(env)
        return env
