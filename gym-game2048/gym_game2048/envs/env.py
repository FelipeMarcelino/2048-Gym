import gym
import numpy as np
import sys
from gym.utils import seeding
from gym import spaces
from .game_2048 import Game2048

class Game2048Env(gym.Env):

    def __init__(self, board_size, seed = None):

        self.state = np.zeros(board_size * board_size)
        self.observation_space = spaces.Discrete(board_size * board_size)
        self.action_space = spaces.Discrete(4) # Up, down, right, left
        self.__game = Game2048(board_size)
        self.__n_iter = False
        self.__done = False
        self.__total_score = 0

    def step(self, action):
        reward = 0
        info = dict()

        print(action)
        sys.exit(1)

        return (self.state, reward, self.__done, info)

    def reset(self):
        self.__n_iter = False
        self.__done = False
        self.__total_score = 0
        self.__game.reset()






