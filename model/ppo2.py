from agent import Agent
from stable_baselines import PPO2
from custom_policy_ppo import CustomCnnPolicy, CustomMlpPolicy
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.evaluation import evaluate_policy
import gym
import gym_game2048
import os
import sys


class PPO2Agent(Agent):
    def __init__(
        self,
        model_name="model_name",
        save_dir="./models",
        log_interval=1e4,
        num_cpus=8,
        eval_episodes=1000,
        n_steps=1e6,
        layer_normalization=False,
        model_kwargs={"tensorboard_log": "./tensorboards/"},
        env_kwargs={"board_size": 4, "binary": True, "extractor": "cnn"},
        callback_checkpoint_kwargs={"save_freq": 0, "save_path": "./models/", "name_prefix": "model_name"},
        callback_hist_kwargs={"hist_freq": 0},
    ):
        super().__init__(
            model_name,
            save_dir,
            num_cpus,
            model_kwargs,
            env_kwargs,
            layer_normalization,
            callback_checkpoint_kwargs,
            callback_hist_kwargs,
            n_steps,
            log_interval,
            eval_episodes,
        )
        self._init_model()

    def _init_model(self):
        if not self._model_kwargs["agent"].lower() == "ppo2":
            raise ValueError(
                "The model_kwargs dict has to be created using args from PPO2 agent as reference. Make sure the correct parameters models."
            )

        del self._model_kwargs["agent"]

        self._callback_checkpoint_kwargs["save_freq"] = int(
            self._callback_checkpoint_kwargs["save_freq"] / self._num_cpus
        )

        if self._env_kwargs["extractor"] == "mlp":
            self._model = PPO2(CustomMlpPolicy, self._env, **self._model_kwargs)
        else:
            self._model = PPO2(CustomCnnPolicy, self._env, **self._model_kwargs)

    def train(self):
        "Optimize the model."
        callbacks = []

        # Checkpoint callback
        if self._callback_checkpoint_kwargs["save_freq"] > 0:

            # Append model name into checkpoint save_path
            self._callback_checkpoint_kwargs["save_path"] = (
                self._callback_checkpoint_kwargs["save_path"] + "/" + str(self._model_name)
            )
            checkpoint_callback = CheckpointCallback(**self._callback_checkpoint_kwargs)
            callbacks.append(checkpoint_callback)

        if self._callback_hist_kwargs["hist_freq"] > 0:
            # hist_callback = CustomCallbackPPO2(**self._callback_hist_kwargs)
            # callbacks.append(hist_callback)
            pass

        try:
            self._model.learn(
                self._n_steps, log_interval=self._log_interval, callback=callbacks, tb_log_name=self._model_name
            )
        except KeyboardInterrupt:
            pass

        folder_path = os.path.join(self._save_dir, self._model_name)
        self._model.save(os.path.join(folder_path, self._model_name))

    def test(self):
        "Evaluate the model."

        mean_reward = super()._test(self._model)

        return mean_reward
