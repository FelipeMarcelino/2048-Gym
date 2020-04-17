import argparse
import optuna
import sys
from ppo2 import PPO2Agent
from acer import ACERAgent
from dqn import DQNAgent


def trial_hiperparameter_dqn(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0005, 1.0),
        "buffer_size": int(trial.suggest_loguniform("buffer_size", 50000, 100000)),
        "exploration_fraction": trial.suggest_loguniform("exploration_fraction", 0.1, 0.3),
        "exploration_final_eps": trial.suggest_loguniform("exploration_final_eps", 0.01, 0.1),
        "exploration_initial_eps": trial.suggest_loguniform("exploration_initial_eps", 0.95, 1.0),
        "train_freq": int(trial.suggest_uniform("train_freq", 1, 8)),
        "batch_size": int(trial.suggest_uniform("batch_size", 32, 128)),
        "target_network_update_freq": int(trial.suggest_uniform("target_network_update_freq", 300, 2000)),
        "prioritized_replay_alpha": trial.suggest_loguniform("prioritized_replay_alpha", 0.5, 0.8),
        "prioritized_replay_beta0": trial.suggest_loguniform("prioritized_replay_beta0", 0.3, 0.6),
    }


def trial_hiperparameter_ppo2(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        "n_steps": int(trial.suggest_loguniform("n_steps", 128, 512)),
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
        "ent_coef": trial.suggest_loguniform("ent_coef", 0.01, 0.1),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.00025, 0.001),
        "vf_coef": trial.suggest_loguniform("vf_coef", 0.4, 0.6),
        "max_grad_norm": trial.suggest_loguniform("max_grad_norm", 0.4, 0.6),
        "lam": trial.suggest_loguniform("lam", 0.9, 1.0),
        "noptepochs": int(trial.suggest_uniform("noptepochs", 2, 10)),
        "cliprange": trial.suggest_loguniform("alpha", 0.2, 0.4),
    }


def trial_hiperparameter_acer(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        "n_steps": int(trial.suggest_loguniform("n_steps", 20, 128)),
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
        "q_coef": trial.suggest_loguniform("q_coef", 0.4, 0.6),
        "ent_coef": trial.suggest_loguniform("ent_coef", 0.01, 0.1),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0007, 0.001),
        "rprop_alpha": trial.suggest_loguniform("rprop_alpha", 0.9, 1.0),
        "rprop_epsilon": trial.suggest_loguniform("rprop_epsilon", 1e-5, 1e-4),
        "buffer_size": int(trial.suggest_loguniform("buffer_size", 5000, 20000)),
        "replay_ratio": trial.suggest_loguniform("replay_ratio", 4, 10),
        "alpha": trial.suggest_loguniform("alpha", 0.9, 1.0000),
        "delta": trial.suggest_loguniform("delta", 0.9, 1.0000),
    }


def optimize_agent(trial, args):

    model_name = args.study_name + "_" + str(trial.number)
    env_kwargs = dict()
    callback_checkpoint_kwargs = dict()
    save_dir = args.save_dir
    log_interval = args.log_interval
    num_cpus = args.num_cpus
    eval_episodes = args.eval_episodes
    n_steps = args.n_steps
    layer_normalization = args.layer_normalization
    layers = args.layers
    env_kwargs["board_size"] = 4
    env_kwargs["binary"] = not args.no_binary
    env_kwargs["extractor"] = args.extractor
    env_kwargs["seed"] = args.seed
    env_kwargs["penalty"] = args.penalty
    callback_checkpoint_kwargs["save_freq"] = args.save_freq
    callback_checkpoint_kwargs["save_path"] = args.save_dir
    callback_checkpoint_kwargs["name_prefix"] = model_name

    if args.agent == "ppo2":
        model_kwargs = trial_hiperparameter_ppo2(trial)
        model_kwargs["agent"] = "ppo2"
        model_kwargs["tensorboard_log"] = args.tensorboard_log
        model = PPO2Agent(
            model_name,
            save_dir,
            log_interval,
            num_cpus,
            eval_episodes,
            n_steps,
            layer_normalization,
            model_kwargs,
            env_kwargs,
            callback_checkpoint_kwargs,
        )
    elif args.agent == "dqn":
        model_kwargs = trial_hiperparameter_dqn(trial)
        model_kwargs["agent"] = "dqn"
        model_kwargs["tensorboard_log"] = args.tensorboard_log
        model_kwargs["double_q"] = True
        model_kwargs["learning_starts"] = 10000
        model_kwargs["prioritized_replay"] = True
        model_kwargs["param_noise"] = True
        model = DQNAgent(
            model_name,
            save_dir,
            log_interval,
            num_cpus,
            eval_episodes,
            n_steps,
            layer_normalization,
            layers,
            model_kwargs,
            env_kwargs,
            callback_checkpoint_kwargs,
        )
    elif args.agent == "acer":
        model_kwargs = trial_hiperparameter_acer(trial)
        model_kwargs["agent"] = "acer"
        model_kwargs["tensorboard_log"] = args.tensorboard_log
        model_kwargs["replay_start"] = 2000
        model = ACERAgent(
            model_name,
            save_dir,
            log_interval,
            num_cpus,
            eval_episodes,
            n_steps,
            layer_normalization,
            model_kwargs,
            env_kwargs,
            callback_checkpoint_kwargs,
        )
    else:
        ValueError("Choose a valid agent model")

    model.train()
    total_score = model.test()

    return total_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        "-ag",
        type=str,
        default="dqn",
        help="Algorithm to use to train the model - DQN, ACER, PPO2",
        required=True,
    )
    parser.add_argument(
        "--tensorboard-log", "-tl", type=str, default="./tensorboards", help="Tensorboard log directory"
    )
    parser.add_argument(
        "--study-name",
        "-sn",
        type=str,
        default="2048-study",
        help="The name of study used for optuna to create \
                        the database.",
    )
    parser.add_argument(
        "--trials",
        "-tr",
        default=0,
        type=int,
        help="The number of trials tested for optuna optimize. - \
                        0 is the default setting and try until the script is finish",
    )
    parser.add_argument(
        "--n-timesteps",
        "-nt",
        type=int,
        dest="n_steps",
        default=10e6,
        help="Number of timestems the model going to run.",
    )
    parser.add_argument("--save-freq", "-sf", type=int, default=10e4, help="The interval between model saves.")
    parser.add_argument("--save-dir", "-sd", type=str, default="./models", help="Save dictory models")
    parser.add_argument("--log-interval", "-li", type=int, default=10e4, help="Log interval")
    parser.add_argument(
        "--no-binary",
        "-bi",
        default=False,
        action="store_true",
        help="Don't use binary \
                        observation space",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--eval-episodes",
        "-ee",
        type=int,
        default=1000,
        help="The number of episodes to test after \
                        training the model",
    )
    parser.add_argument(
        "--extractor",
        "-ex",
        type=str,
        default="cnn",
        help="The extractor used to create the features \
                        from observation space - (mlp or cnn)",
    )
    parser.add_argument(
        "--layer-normalization",
        "-ln",
        default=False,
        action="store_true",
        help="Use layer normalization \
        - Only for DQN",
    )
    parser.add_argument("--num-cpus", "-nc", default=1, type=int, help="Number of cpus to use. DQN only accept 1")
    parser.add_argument(
        "--layers",
        "-l",
        nargs="+",
        type=int,
        default=[64, 64],
        help="List of neurons to use in DQN algorithm. \
                        The number of elements inside list going to be the number of layers.",
    )
    parser.add_argument(
        "--penalty", "-pe", default=-512, type=int, help="How much penalize the model when choose a invalid action"
    )

    args = parser.parse_args()
    study = optuna.create_study(
        storage="postgresql://felipemarcelino:123456@localhost/database",
        direction="maximize",
        load_if_exists=True,
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(),
    )

    trials = None
    if args.trials != 0:
        trials = args.trials
    study.optimize(lambda trial: optimize_agent(trial, args), n_trials=trials, n_jobs=1)


if __name__ == "__main__":
    main()
