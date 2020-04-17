from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import os


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, log_dir="logs", hist_freq: int = 100, verbose=0, log_file="log", eval_episodes=100):
        super(CustomCallback, self).__init__(verbose)
        self.num_episodes = 1
        self.max_val = 0
        self.histogram = np.zeros(15, dtype=int)
        self.verbose = verbose
        self.hist_freq = int(hist_freq)
        self.log_dir = log_dir
        self.log_file = log_file
        self.eval_episodes = eval_episodes
        self.last_timestep = 1
        self.episode_lengths = []
        self.episode_maxtiles = []
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        # print(self.verbose, type(self.hist_freq))
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        timestep = self.locals["self"].num_timesteps
        num_episodes = len(self.locals["episode_rewards"])
        if num_episodes > self.num_episodes:
            self.num_episodes = num_episodes
            self.histogram[self.max_val] += 1
            self.episode_maxtiles.append(self.max_val)
            self.episode_lengths.append(timestep - self.last_timestep)
            self.last_timestep = timestep
            if self.hist_freq > 0 and num_episodes % self.hist_freq == 0:
                self._dump_values()
        self.max_val = self.locals["self"].get_env().maximum_tile()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _dump_values(self):
        timestep = self.locals["self"].num_timesteps
        num_episodes = self.num_episodes
        if self.log_dir and self.log_file:
            log_path = os.path.join(self.log_dir, self.log_file)
            log_file = log_path + ".npz"
            try:
                os.replace(log_file, log_file + ".bkp")
            except:
                pass
            np.savez(
                log_path,
                rewards=self.locals["episode_rewards"],
                lengths=self.episode_lengths,
                max_tiles=self.episode_maxtiles,
            )

        if self.verbose:
            print()
            print(f"#episodes: {num_episodes}")
            print(f"#timesteps: {timestep}")
            print(f'Mean rewards: {np.mean(self.locals["episode_rewards"][-self.eval_episodes:])}')
            print(f"Mean episode length: {np.mean(self.episode_lengths[-self.eval_episodes:])}")
            print("Histogram of maximum tile achieved:")
            for i in range(1, 15):
                if self.histogram[i] > 0:
                    print(f"{2**i}: {self.histogram[i]}")
            print()


class CustomCallbackPPO2(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, log_dir="logs", hist_freq: int = 100, verbose=0, log_file="log", eval_episodes=100):
        super(CustomCallbackPPO2, self).__init__(verbose)
        self.num_episodes = 1
        self.max_val = 0
        self.histogram = np.zeros(15, dtype=int)
        self.verbose = verbose
        self.hist_freq = int(hist_freq)
        self.log_dir = log_dir
        self.log_file = log_file
        self.eval_episodes = eval_episodes
        self.last_timestep = 1
        # self.episode_lengths= []
        self.episode_maxtiles = []
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        # print(self.verbose, type(self.hist_freq))
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        timestep = self.locals["self"].num_timesteps
        env = self.training_env
        episode_rewards = env.get_attr("episode_rewards")
        n_env = len(episode_rewards)
        num_episodes = 0
        for l in episode_rewards:
            num_episodes += len(l)
        if num_episodes > self.num_episodes:
            self.num_episodes = num_episodes
            self.histogram[self.max_val] += 1
            self.episode_maxtiles.append(self.max_val)
            # self.episode_lengths.append(timestep-self.last_timestep)
            self.last_timestep = timestep
            if self.hist_freq > 0 and num_episodes % self.hist_freq == 0:
                self._dump_values()
        self.max_val = env.env_method("maximum_tile")
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _dump_values(self):
        timestep = self.locals["self"].num_timesteps
        num_episodes = self.num_episodes
        env = self.training_env
        episode_rewards = env.get_attr("episode_rewards")
        if self.log_dir and self.log_file:
            log_path = os.path.join(self.log_dir, self.log_file)
            log_file = log_path + ".npz"
            try:
                os.replace(log_file, log_file + ".bkp")
            except:
                pass
            np.savez(log_path, rewards=episode_rewards, max_tiles=self.episode_maxtiles)

        if self.verbose:
            n_env = len(episode_rewards)
            print()
            print(f"#episodes: {num_episodes}")
            print(f"#timesteps: {timestep}")
            mean_rewards = 0
            for l in episode_rewards:
                mean_rewards += np.mean(l[-self.eval_episodes // n_env :]) / n_env
            print(f"Mean rewards: {mean_rewards}")
            # print(f'Mean episode length: {np.mean(self.episode_lengths[-self.eval_episodes:])}')
            print("Histogram of maximum tile achieved:")
            for i in range(1, 15):
                if self.histogram[i] > 0:
                    print(f"{2**i}: {self.histogram[i]}")
            print()

