import gym
from abc import ABCMeta, abstractmethod
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
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
    ):
        """
        Agent base class. This class implement common functions and variables for all agents(dqn,ppo,acer).

        Parameters
        ----------
        model_name : str
            Model name.
        save_dir : str
            Directory to save model.
        num_cpus : int
            Number of cpus to run the environment.
        model_kwargs : dict
            Model parameters.
        env_kwargs : dict
            Env parameters.
        layer_normalization : bool
            Adding layer normalization on MLP or CNN.
        callback_checkpoint_kwargs : dict
            Parameters of callback.
        callback_hist_kwargs : dict
            Parameters of callback hist(NOT Implemented.)
        n_steps : int
            How much steps to run the model.
        log_interval : int
            Interval steps between log output.
        eval_episodes : int
            How much episodes evaluate the policy.
        load_path : str
            Load model from.
        num_timesteps : int
            Continuing log using num_ti .

        """
        self._model_kwargs = model_kwargs
        self._env_kwargs = env_kwargs
        self._save_dir = save_dir
        self._model_name = model_name
        self._num_cpus = num_cpus
        self._layer_normalization = layer_normalization
        self._callback_checkpoint_kwargs = callback_checkpoint_kwargs
        self._callback_hist_kwargs = callback_hist_kwargs
        self._n_steps = n_steps
        self._log_interval = log_interval
        self._eval_episodes = eval_episodes
        self._load_path = load_path
        self._num_timesteps = num_timesteps
        env = gym.make("gym_game2048:game2048-v0", **self._env_kwargs)
        self._env = DummyVecEnv([lambda: env for i in range(self._num_cpus)])

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def _test(self, model):

        env = gym.make("gym_game2048:game2048-v0", **self._env_kwargs)
        self._env = DummyVecEnv([lambda: env for i in range(1)])

        mean_reward, _ = evaluate_policy(model, self._env, self._eval_episodes, deterministic=True)

        return mean_reward
