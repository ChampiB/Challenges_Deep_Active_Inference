import gym


#
# Class picking the FIRE action in environment requiring it.
#
class FireResetWrapper(gym.Wrapper):

    def __init__(self, env=None):
        """
        Constructor.
        :param env: the environment to wrap.
        """
        super().__init__(env)
        if env.unwrapped.get_action_meanings()[1] != 'FIRE':
            raise RuntimeError("FireResetWrapper should not be applied to this environment.")
        if len(env.unwrapped.get_action_meanings()) < 3:
            raise RuntimeError("FireResetWrapper should not be applied to this environment.")

    def step(self, action):
        """
        Execute the action ask by the agent.
        :param action: the action to execute.
        :return: the observation received when taking the action.
        """
        return self.env.step(action)

    def reset(self):
        """
        Pick the FIRE action in when the environment just got reset
        :return: the observation received by the agent after the reset.
        """
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

