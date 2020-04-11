import gym
from abc import ABCMeta, abstractmethod
from stable_baselines.common.vec_env import DummyVecEnv


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self, env_name, num_cpus, model_kwargs, env_kwargs, custom_policy, callback_kwargs, n_steps, log_interval
    ):

        self._model_kwargs = model_kwargs
        self._env_kwargs = env_kwargs
        self._env_name = env_name
        self._num_cpus = num_cpus
        self._custom_policy = custom_policy
        self._callback_kwargs = callback_kwargs
        self._n_steps = n_steps
        self._log_interval = log_interval
        env = gym.make(self._env_name, **self._env_kwargs)
        self._env = DummyVecEnv([lambda: env for i in range(self._num_cpus)])

    @abstractmethod
    def _init_model(self):
        pass
