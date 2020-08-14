from agent import Agent
from stable_baselines import DQN
from custom_policy_dqn import CustomCnnPolicy, CustomMlpPolicy, CustomLnCnnPolicy, CustomLnMlpPolicy
from stable_baselines.common.callbacks import CheckpointCallback
import gym
import gym_game2048
import os


class DQNAgent(Agent):
    def __init__(
        self,
        model_name="model_name",
        save_dir="./models",
        log_interval=1e4,
        num_cpus=1,
        eval_episodes=1000,
        n_steps=1e6,
        layer_normalization=False,
        layers=[64, 64],
        load_path=None,
        num_timesteps=None,
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
            load_path,
            num_timesteps,
        )
        self.__layers = layers
        self._init_model()

    def _init_model(self):
        if not self._model_kwargs["agent"].lower() == "dqn":
            raise ValueError(
                "The model_kwargs dict has to be created using args from DQN agent as reference. Make sure the correct parameters models."
            )

        del self._model_kwargs["agent"]

        if self._load_path is not None and self._num_timesteps is not None:
            print("Loading model...", self._load_path)
            self._model = DQN.load(
                self._load_path,
                num_timesteps=self._num_timesteps,
                env=self._env,
                tensorboard_log=self._model_kwargs["tensorboard_log"],
            )

        if self._env_kwargs["extractor"] == "mlp":
            if self._layer_normalization is True:
                self._model = DQN(
                    CustomLnMlpPolicy, self._env, policy_kwargs={"layers": self.__layers}, **self._model_kwargs
                )
            else:
                self._model = DQN(
                    CustomMlpPolicy, self._env, policy_kwargs={"layers": self.__layers}, **self._model_kwargs
                )
        else:
            if self._layer_normalization is True:
                self._model = DQN(
                    CustomLnCnnPolicy, self._env, policy_kwargs={"layers": self.__layers}, **self._model_kwargs
                )
            else:
                self._model = DQN(
                    CustomCnnPolicy, self._env, policy_kwargs={"layers": self.__layers}, **self._model_kwargs
                )

    def train(self):
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
            # hist_callback = CustomCallback(**self._callback_hist_kwargs)
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

        mean_reward = super()._test(self._model)
        return mean_reward
