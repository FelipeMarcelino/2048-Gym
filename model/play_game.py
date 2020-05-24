import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
import time
import pickle


import gym
import gym_game2048
from stable_baselines import PPO2, results_plotter, DQN, ACER
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.callbacks import CheckpointCallback
from custom_policy_dqn import CustomCnnPolicy, CustomMlpPolicy, CustomLnCnnPolicy, CustomLnMlpPolicy
import tensorflow.contrib.layers as tf_layers

from tqdm.auto import tqdm

from stable_baselines.deepq.policies import DQNPolicy, CnnPolicy
from stable_baselines.bench import Monitor

from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0


def create_env(board_size, seed, binary):

    env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary, extractor="cnn", penalty=0)

    return env


def main():

    env = create_env(4, None, True)

    model = DQN.load("models/no_penalty_custom_parameters_3_4/no_penalty_custom_3_4.zip")

    total_score = 0
    total_2048 = 0
    best_score_episode = 0
    best_actions_episode = None
    best_boards_episode = None
    best_actions_episode_without_zero = None
    best_boards_episode_without_zero = None
    best_episode = {}
    best_without_zero_episode = {}

    for i in tqdm(range(1000)):

        # obs = game.reset()
        obs = env.reset()
        done = False
        total_score_episode = 0
        actions = []
        boards = []
        actions_without_zero = []
        boards_without_zero = []

        while done is False:
            action, states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            last_board = env.get_board()
            total_score_episode = info["total_score"]
            actions.append(action)
            boards.append(env.get_board())
            actions_without_zero.append(action)
            boards_without_zero.append(env.get_board())

        last_board = env.get_board()

        if last_board.max() == 2048:
            total_2048 = total_2048 + 1
            if total_score_episode > best_score_episode:
                best_score_episode = total_score_episode
                best_actions_episode = actions
                best_boards_episode = boards
            if last_board.min() != 0:
                best_actions_episode_without_zero = actions_without_zero
                best_boards_episode_without_zero = boards_without_zero

        total_score = total_score + total_score_episode

    print("Mean:", total_score / 1000)
    print("Wins:", total_2048)
    best_episode["actions"] = best_actions_episode
    best_episode["boards"] = best_boards_episode
    best_episode["score"] = best_score_episode
    with open("best_episode.pkl", "wb") as pickle_file:
        pickle.dump(best_episode, pickle_file, pickle.HIGHEST_PROTOCOL)

    if best_actions_episode_without_zero is not None:
        best_without_zero_episode["actions"] = best_actions_episode_without_zero
        best_without_zero_episode["boards"] = best_boards_episode_without_zero

        with open("best_episode_without_zero.pkl", "wb") as pickle_file:
            pickle.dump(best_without_zero_episode, pickle_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
