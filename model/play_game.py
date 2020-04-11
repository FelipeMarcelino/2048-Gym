import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
import time

import tkinter as tk

import gym
import gym_game2048
from stable_baselines import PPO2, results_plotter, DQN, ACER
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.callbacks import CheckpointCallback
import tensorflow.contrib.layers as tf_layers
from gym_game2048.envs.game_grid import GameGrid

from tqdm.auto import tqdm

# from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy, CnnPolicy
from stable_baselines.deepq.policies import DQNPolicy, CnnPolicy
from stable_baselines.bench import Monitor

# from stable_baselines.deepq import MlpPolicy
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0


f1 = tk.Frame(width=600, height=600)


def create_env(board_size, seed, binary):

    env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary)
    # env = gym.make("game2048-macenta-v0", size=board_size, seed=seed)

    return env


def main():

    env = create_env(4, None, True)
    # game = GameGrid(4)
    # game.update()

    model = DQN.load("modelos/20-million-dqn_10950000_steps.zip")

    total_score = 0

    for i in range(100):

        # obs = game.reset()
        obs = env.reset()
        done = False
        total_score_episode = 0

        while done is False:
            action, states = model.predict(obs, deterministic=True)
            # obs, rewards, done, info = game.step(action)
            obs, rewards, done, info = env.step(action)
            last_board = env.get_board()
            total_score_episode = info["total_score"]

        last_board = env.get_board()

        if last_board.max() == 2048:
            print(last_board.reshape((4, 4)))
            print("Next...")
            time.sleep(5)

        total_score = total_score + total_score_episode

    print(total_score / 100)


if __name__ == "__main__":
    main()
