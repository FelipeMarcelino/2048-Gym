import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf

import gym
import gym_game2048
from stable_baselines import PPO2, results_plotter, DQN, ACER, HER
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.callbacks import CheckpointCallback
import tensorflow.contrib.layers as tf_layers

from tqdm.auto import tqdm

# from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, register_policy, CnnPolicy

from stable_baselines.deepq.policies import DQNPolicy, CnnPolicy
from stable_baselines.bench import Monitor

# from stable_baselines.deepq import MlpPolicy
from stable_baselines.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0

study_name = sys.argv[1]

control = {0: "up", 1: "down", 2: "right", 3: "left"}


def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    # layer 1
    conv1 = conv(scaled_images, "c1", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv2 = conv(scaled_images, "c2", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    relu1 = activ(conv1)
    relu2 = activ(conv2)
    # layer 2
    conv11 = conv(relu1, "c3", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv12 = conv(relu1, "c4", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv21 = conv(relu2, "c3", n_filters=128, filter_size=(1, 2), stride=1, init_scale=np.sqrt(2), **kwargs)
    conv22 = conv(relu2, "c4", n_filters=128, filter_size=(2, 1), stride=1, init_scale=np.sqrt(2), **kwargs)
    # layer2 relu activation
    relu11 = tf.nn.relu(conv11)
    relu12 = tf.nn.relu(conv12)
    relu21 = tf.nn.relu(conv21)
    relu22 = tf.nn.relu(conv22)

    # get shapes of all activations
    shape1 = relu1.get_shape().as_list()
    shape2 = relu2.get_shape().as_list()

    shape11 = relu11.get_shape().as_list()
    shape12 = relu12.get_shape().as_list()
    shape21 = relu21.get_shape().as_list()
    shape22 = relu22.get_shape().as_list()

    # expansion
    hidden1 = tf.reshape(relu1, [-1, shape1[1] * shape1[2] * shape1[3]])
    hidden2 = tf.reshape(relu2, [-1, shape2[1] * shape2[2] * shape2[3]])

    hidden11 = tf.reshape(relu11, [-1, shape11[1] * shape11[2] * shape11[3]])
    hidden12 = tf.reshape(relu12, [-1, shape12[1] * shape12[2] * shape12[3]])
    hidden21 = tf.reshape(relu21, [-1, shape21[1] * shape21[2] * shape21[3]])
    hidden22 = tf.reshape(relu22, [-1, shape22[1] * shape22[2] * shape22[3]])

    # concatenation
    hidden = tf.concat([hidden1, hidden2, hidden11, hidden12, hidden21, hidden22], axis=1)

    # linear layer 1
    linear_1 = activ(linear(hidden, scope="fc1", n_hidden=512, init_scale=np.sqrt(2)))

    # linear layer 2
    linear_2 = activ(linear(hidden, scope="fc2", n_hidden=128, init_scale=np.sqrt(2)))

    return linear_2


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, "c1", n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, "c2", n_filters=64, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, "c3", n_filters=64, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, "fc1", n_hidden=512, init_scale=np.sqrt(2)))


# Custom MLP policy of three layers of size 128 each
#  class CustomPolicy(FeedForwardPolicy):
#      def __init__(self, *args, **kwargs):
#          super(CustomPolicy, self).__init__(
#              *args, **kwargs, net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])], feature_extraction="mlp"
#          )


#  class CustomPolicy(CnnPolicy):
#      def __init__(self, *args, **kwargs):
#          super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_cnn)


class CustomPolicy(DQNPolicy):
    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        reuse=True,
        layers=None,
        extractor="",
        feature_extraction="cnn",
        obs_phs=None,
        layer_norm=False,
        dueling=True,
        act_fun=tf.nn.relu,
        **kwargs
    ):
        super(CustomPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            dueling=dueling,
            reuse=reuse,
            scale=(feature_extraction == "cnn"),
            obs_phs=obs_phs,
        )

        self._kwargs_check(feature_extraction, kwargs)

        cnn_extractor = modified_cnn

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("action_value"):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    action_out = extracted_features
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    action_out = extracted_features
                for layer_size in layers:
                    action_out = tf_layers.fully_connected(action_out, num_outputs=layer_size, activation_fn=None)
                    if layer_norm:
                        action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                    action_out = act_fun(action_out)

                action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)

            if self.dueling:
                with tf.variable_scope("state_value"):
                    state_out = extracted_features
                    for layer_size in layers:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=layer_size, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


#  def callback(_locals, _globals):
#      """
#      callback called at each step (for dqn an others) or after n steps (see acer or ppo2)
#      :param _locals: (dict)
#      :param _globals: (dict)
#      """
#      global n_steps, best_mean_reward
#      # print stats every 1000 calls
#      if (n_steps + 1) % 1000 == 0:
#          # evaluate policy training performance
#          x, y = ts2xy(load_results(log_dir), "timesteps")
#          if len(x) > 0:
#              mean_reward = np.mean(y[-100:])
#              print(x[-1], "timesteps")
#              print(
#                  "best mean reward: {:.2f} - last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward)
#              )
#
#              # new best model, you could save the agent here
#              if mean_reward > best_mean_reward:
#                  best_mean_reward = mean_reward
#                  # example for saving best model
#                  print("saving new best model")
#                  _locals["self"].save(log_dir + "best_model.pkl")
#      n_steps += 1
#      return True


class TqdmCallback(object):
    def __init__(self):
        self.pbar = None

    def __call__(self, _locals, _globals):
        if self.pbar is None:
            self.pbar = tqdm(total=_locals["_"])

        self.pbar.update(1)

        if _locals["total_timesteps"] == _locals["_"]:
            self.pbar.close()
            self.pbar = None

        return True


tqdm_callback = TqdmCallback()

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


def create_env(board_size, seed, binary):

    env = gym.make("game2048-v0", board_size=board_size, seed=seed, binary=binary)
    # env = gym.make("game2048-macenta-v0", size=board_size, seed=seed)

    return env


def train_model(env, steps, model_params):

    env = Monitor(env, log_dir, allow_early_resets=True)

    num_cpu = 1
    # env = DummyVecEnv([lambda: env for i in range(num_cpu)])

    # model = PPO2(MlpPolicy, env, verbose=0, seed=1, tensorboard_log="./tensorboards/")
    model = DQN(
        CustomPolicy,
        env,
        prioritized_replay=True,
        buffer_size=10000,
        learning_rate=1e-3,
        learning_starts=10000,
        target_network_update_freq=1000,
        train_freq=4,
        exploration_final_eps=0.01,
        exploration_fraction=0.1,
        prioritized_replay_alpha=0.6,
        verbose=0,
        tensorboard_log="./tensorboards/",
    )
    # model = ACER(MlpPolicy, env, verbose=0, **model_params)
    # model = ACER(MlpPolicy, env, verbose=0, tensorboard_log="./tensorboards/", **model_params)
    # model = ACER(CustomPolicy, env, verbose=0, seed=1, tensorboard_log="./tensorboards/", **model_params)
    # model = PPO2(CustomPolicy, env, verbose=0, seed=1, tensorboard_log="./tensorboards/", **model_params)

    #  model = ACER(
    #      CustomPolicy,
    #      env,
    #      buffer_size=20000,
    #      learning_rate=1e-4,
    #      replay_start=10000,
    #      replay_ratio=4,
    #      verbose=0,
    #      tensorboard_log="./tensorboards/",
    #  )

    # goal_selection_strategy = "future"  # equivalent to GoalSelectionStrategy.FUTURE
    # model = HER(CustomPolicy, env, DQN, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, verbose=1)

    # model.load_parameters("./modelos/20-million-dqn_10950000_steps.zip")
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path="./modelos/", name_prefix="20-million-acer")
    model.learn(total_timesteps=steps, callback=checkpoint_callback)

    results_plotter.plot_results([log_dir], steps, results_plotter.X_TIMESTEPS, "ACER psr")
    plt.savefig("20-million-acer.png")
    model.save("20-million-acer")

    return model


def test_model(model, env):
    total_score = 0

    for i in range(1000):

        obs = env.reset()
        done = false
        total_score_episode = 0

        while done is false:
            action, states = model.predict(obs, deterministic=true)
            obs, rewards, done, info = env.step(action)
            total_score_episode = info["total_score"]

        total_score = total_score + total_score_episode
        # print("last action:", control[action])
        # print("before move:\n", info["before_move"].reshape((4, 4)))
        # print("last obs:\n", obs.reshape((4, 4)))
        # print(rewards)

    return total_score / 1000


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


def set_params():
    """ Learning hyperparamters we want to optimise"""
    return {
        #  "n_steps": int(100),
        #  "gamma": 0.98,
        #  "learning_rate": 0.0001,
        #  "rprop_alpha": 0.02,
        #  "rprop_epsilon": 1e-4,
        #  "ent_coef": 1e-4,
        #  "replay_start": int(20000),
        #  "alpha": 0.92,
        #  "delta": 0.95,
        #  "q_coef": 0.4,
    }


def optimize_agent(trial):

    model_params = trial_hiperparameter(trial)
    env = create_env(4, None, True)
    model = train_model(env, 20000000, model_params)
    total_score = test_model(model, env)

    return total_score


def train_specific_params():

    env = create_env(4, None, True)
    model_params = set_params()
    model = train_model(env, 20000000, model_params)
    total_score = test_model(model, env)
    print(total_score)


def main():

    #  study = optuna.create_study(
    #      storage="postgresql://felipemarcelino:123456@localhost/database",
    #      direction="maximize",
    #      load_if_exists=True,
    #      study_name=study_name,
    #      pruner=optuna.pruners.MedianPruner(),
    #  )
    #  study.optimize(optimize_agent, n_trials=None, n_jobs=1)
    train_specific_params()


if __name__ == "__main__":
    main()
