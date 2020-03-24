import gym
import gym_game2048
from stable_baselines.common.env_checker import check_env

board_size = 4
seed = None
env = gym.make("game2048-v0", board_size=board_size, seed=seed)
check_env(env)
