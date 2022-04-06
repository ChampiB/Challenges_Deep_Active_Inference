from gym.utils import play
import gym

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    gym.utils.play.play(env, zoom=5, fps=10)
