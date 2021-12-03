from environments.wrappers.FireResetWrapper import FireResetWrapper
from environments.wrappers.NormalizePixelsWrapper import NormalizePixelsWrapper
from environments.wrappers.ToPyTorchTensorWrapper import ToPyTorchTensorWrapper
from environments.wrappers.ProcessFrameWrapper import ProcessFrameWrapper
from environments.wrappers.BufferWrapper import BufferWrapper
from singletons.Device import Device
import gym


#
# Class applying all the default wrappers to an environment.
#
class DefaultWrappers:

    @staticmethod
    def apply(env, image_shape):
        """
        Apply all the default wrapper to the environment.
        :param env: the environment to wrap.
        :param image_shape: the shape of the input image.
        :return: the wrapped environment.
        """
        # Only support discrete actions.
        assert isinstance(env.action_space, gym.spaces.Discrete)

        # Apply images wrapper if the environment produces images and non-images wrappers otherwise.
        if DefaultWrappers.env_returns_images(env):
            return DefaultWrappers.apply_images_wrappers(env, image_shape)
        else:
            return DefaultWrappers.apply_non_images_wrappers(env)

    @staticmethod
    def env_returns_images(env):
        """
        Check if the environment returns images.
        :param env: the environment to check.
        :return: true if the environment returns images, false otherwise.
        """
        return isinstance(env.observation_space, gym.spaces.box.Box) \
            and len(env.observation_space.shape) >= 2

    @staticmethod
    def apply_images_wrappers(env, image_shape):
        """
        Apply all the default wrapper to the environment.
        :param env: the environment to wrap.
        :param image_shape: the shape of the input image.
        :return: the wrapped environment.
        """

        # Apply the default wrappers.
        try:
            env = FireResetWrapper(env)
        except (RuntimeError, AttributeError):
            print("FireResetWrapper was not applied to the environment.")
        env = ProcessFrameWrapper(env, image_shape)
        env = BufferWrapper(env, image_shape[0])
        env = NormalizePixelsWrapper(env)
        env = ToPyTorchTensorWrapper(env, Device.get())
        return env

    @staticmethod
    def apply_non_images_wrappers(env):
        """
        Apply all the default wrapper to the environment.
        :param env: the environment to wrap.
        :return: the wrapped environment.
        """

        # Apply the default wrappers.
        try:
            env = FireResetWrapper(env)
        except (RuntimeError, AttributeError):
            print("FireResetWrapper was not applied to the environment.")
        env = ToPyTorchTensorWrapper(env, Device.get())
        return env
