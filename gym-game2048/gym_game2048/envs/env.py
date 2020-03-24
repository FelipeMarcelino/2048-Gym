import gym
import numpy as np
import sys
from gym.utils import seeding
from gym import spaces
from .game_2048 import Game2048


class Game2048Env(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size, seed=None):

        self.state = np.zeros(board_size * board_size)
        self.observation_space = spaces.Box(0, 2 ** 16, (board_size * board_size,), dtype=np.uint32)
        self.action_space = spaces.Discrete(4)  # Up, down, right, left
        self.__game = Game2048(board_size)
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__board_size = board_size

    def step(self, action):
        reward = 0
        info = dict()

        # print("Action:", action)
        # print(self.__game.get_board())
        self.__game.make_move(action)
        self.__game.confirm_move()
        self.state = self.__game.get_board().flatten()
        self.__done = self.__game.verify_game_state()
        reward = self.__game.get_move_score()
        # print(self.__game.get_board())
        # print(reward)
        # sys.exit(1)
        self.__n_iter = self.__n_iter + 1

        info["total_score"] = self.__game.get_total_score()

        return (self.state, reward, self.__done, info)

    def reset(self):
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__game.reset()
        self.state = self.__game.get_board().flatten()
        return self.state

    def render(self, mode="human"):
        pass

