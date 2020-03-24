import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna

import gym
import gym_game2048
from stable_baselines import PPO2, results_plotter, DQN, ACER
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy
from stable_baselines.bench import Monitor

# from stable_baselines.deepq import MlpPolicy
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0

study_name = sys.argv[1]


def callback(_locals, _globals):
    """
    callback called at each step (for dqn an others) or after n steps (see acer or ppo2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], "timesteps")
            print(
                "best mean reward: {:.2f} - last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward)
            )

            # new best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # example for saving best model
                print("saving new best model")
                _locals["self"].save(log_dir + "best_model.pkl")
    n_steps += 1
    return True


log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


def create_env(board_size, seed):

    env = gym.make("game2048-v0", board_size=board_size, seed=seed)

    return env


def train_model(env, steps, model_params):

    env = Monitor(env, log_dir, allow_early_resets=True)

    num_cpu = 8
    env = DummyVecEnv([lambda: env for i in range(num_cpu)])

    # model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log="./tensorboards/")
    # model = DQN(MlpPolicy, env, verbose=0, tensorboard_log="./tensorboards/")
    model = ACER(MlpPolicy, env, verbose=0, tensorboard_log="./tensorboards/", **model_params)
    model.learn(total_timesteps=steps)

    # results_plotter.plot_results([log_dir], steps, results_plotter.X_TIMESTEPS, "PPO2 psr")
    # plt.savefig("first_model.png")
    # model.save("first_model")

    return model


def test_model(model, env):
    total_score = 0

    for i in range(10000):

        obs = env.reset()
        done = False
        total_score_episode = 0

        while done is False:
            action, states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_score_episode = info["total_score"]

        total_score = total_score + total_score_episode

    return total_score / 10000


def trial_hiperparameter(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        "n_steps": int(trial.suggest_loguniform("n_steps", 64, 256)),
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
        "rprop_alpha": trial.suggest_loguniform("rprop_alpha", 0.001, 1.0),
        "rprop_epsilon": trial.suggest_loguniform("rprop_epsilon", 1e-6, 1e-3),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
        "replay_start": int(trial.suggest_uniform("replay_start", 10000, 40000)),
        "alpha": trial.suggest_loguniform("alpha", 0.9, 0.9999),
        "delta": trial.suggest_loguniform("delta", 0.9, 1.0000),
        "q_coef": trial.suggest_loguniform("q_coef", 0.3, 0.6),
    }


def optimize_agent(trial):

    model_params = trial_hiperparameter(trial)
    env = create_env(4, None)
    model = train_model(env, 400000, model_params)
    total_score = test_model(model, env)

    return total_score


def main():

    study = optuna.create_study(
        storage="postgresql://felipemarcelino:123456@localhost/database",
        direction="maximize",
        load_if_exists=True,
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(optimize_agent, n_trials=None, n_jobs=1)


if __name__ == "__main__":
    main()
