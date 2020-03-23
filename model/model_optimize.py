import sys
import os

import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy


def create_env(board_size, seed):

    env = gym.make("game2048-gym", board_size=board_size, seed=seed)

    return env

def train_model(env, steps):

    num_cpu = 8
    env = DummyVecEnv([lambda: env for i in range(num_cpu)])

    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./tensorboards/")
    model.learn(total_timesteps=steps)

    return model


def main():

    env = create_env(4, None)
    train_model(env, 200000)

if __name__ == "__main__":
    main()
