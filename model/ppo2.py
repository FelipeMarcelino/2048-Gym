from agents import Agent
from stable_baselines import PPO2
from custom_policy_ppo import CustomCnnPolicy, CustomMlpPolicy
from stable_baselines.common.callbacks import CheckpointCallback
import gym
import gym_game2048


class PPO2Agent(Agent):
    def __init__(
        self,
        env_name="gym_game2048:game2048-v0",
        log_interval=1e4,
        num_cpus=8,
        custom_policy=None,
        n_steps=1e6,
        model_kwargs={"tensorboard_log": "./tensorboards/"},
        env_kwargs={"board_size": 4, "binary": True, "mlp": False},
        callbacks_kwargs={"save_freq": 0},
    ):
        super().__init__(
            env_name, num_cpus, model_kwargs, env_kwargs, custom_policy, callbacks_kwargs, n_steps, log_interval
        )
        self._init_model()

    def _init_model(self):
        if not self._model_kwargs["agent"].lower() == "ppo2":
            raise ValueError(
                "The model_kwargs dict has to be created using args from PPO2 agent as reference. Make sure the correct parameters models."
            )

        del self._model_kwargs["agent"]

        if self._env_kwargs["mlp"] is False:
            self._model = PPO2(CustomCnnPolicy, self._env, **self._model_kwargs)
        else:
            self._model = PPO2(CustomMlpPolicy, self._env, **self._model_kwargs)

    def train(self):
        callbacks = []

        # Checkpoint callback
        if self._callback_kwargs["save_freq"] > 0:
            checkpoint_callback = CheckpointCallback(**self._model_kwargs)
            callbacks.append(checkpoint_callback)

        self._model.learn(self._n_steps, log_interval=self._log_interval, callback=callbacks)


var_env_kwargs = {"board_size": 4, "binary": True, "mlp": True}
ppo2_agent = PPO2Agent(model_kwargs={"agent": "ppo2", "gamma": 0.98}, n_steps=100000, env_kwargs=var_env_kwargs)
ppo2_agent.train()
