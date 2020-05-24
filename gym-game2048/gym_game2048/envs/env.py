import gym
import numpy as np
import sys
from gym.utils import seeding
from gym import spaces
from .game_2048 import Game2048


class Game2048Env(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, board_size, binary, extractor, invalid_move_warmup=16, invalid_move_threshold=0.1, penalty=-512, seed=None
    ):
        """
        Iniatialize the environment.


        Parameters
        ----------
        board_size : int
            Size of the board. Default=4
        binary : bool
            Use binary representation of the board(power 2 matrix). Default=True
        extractor : str
            Type of model to extract the features. Default=cnn
        invalid_move_warmup : int
            Minimum of invalid movements to finish the game. Default=16
        invalid_move_threshold : float
                    How much(fraction) invalid movements is necessary according to the total of moviments already executed. to finish the episode after invalid_move_warmup. Default 0.1
        penalty : int
            Penalization score of invalid movements to sum up in reward function. Default=-512
        seed :  int
            Seed
        """

        self.state = np.zeros(board_size * board_size)
        self.__binary = binary
        self.__extractor = extractor

        if self.__binary is True:
            if extractor == "cnn":
                self.observation_space = spaces.Box(
                    0, 1, (board_size, board_size, 16 + (board_size - 4)), dtype=np.uint32
                )
            else:
                self.observation_space = spaces.Box(
                    0, 1, (board_size * board_size * (16 + (board_size - 4)),), dtype=np.uint32
                )
        else:
            if extractor == "mlp":
                self.observation_space = spaces.Box(0, 2 ** 16, (board_size * board_size,), dtype=np.uint32)
            else:
                ValueError("Extractor must to be mlp when observation space is not binary")

        self.action_space = spaces.Discrete(4)  # Up, down, right, left

        if penalty > 0:
            raise ValueError("The value of penalty needs to be between [0, -inf)")
        self.__game = Game2048(board_size, invalid_move_warmup, invalid_move_threshold, penalty)
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__board_size = board_size

    def step(self, action):
        """
        Execute a action.

        Parameters
        ----------
        action : int
            Action selected by the model.
        """
        reward = 0
        info = dict()

        before_move = self.__game.get_board().copy()
        self.__game.make_move(action)
        self.__game.confirm_move()

        if self.__binary is True:
            self.__game.transform_board_to_power_2_mat()

            if self.__extractor == "cnn":
                self.state = self.__game.get_power_2_mat()
            else:
                self.state = self.__game.get_power_2_mat().flatten()
        else:
            self.state = self.__game.get_board().flatten()

        self.__done, penalty = self.__game.verify_game_state()
        reward = self.__game.get_move_score() + penalty
        self.__n_iter = self.__n_iter + 1
        after_move = self.__game.get_board()

        info["total_score"] = self.__game.get_total_score()
        info["steps_duration"] = self.__n_iter
        info["before_move"] = before_move
        info["after_move"] = after_move

        return (self.state, reward, self.__done, info)

    def reset(self):
        "Reset the environment."
        self.__n_iter = 0
        self.__done = False
        self.__total_score = 0
        self.__game.reset()
        if self.__binary is True:
            self.__game.transform_board_to_power_2_mat()
            if self.__extractor == "cnn":
                self.state = self.__game.get_power_2_mat()
            else:
                self.state = self.__game.get_power_2_mat().flatten()
        else:
            self.state = self.__game.get_board().flatten()
        return self.state

    def render(self, mode="human"):
        pass

    def get_board(self):
        "Get the board."

        return self.__game.get_board()

