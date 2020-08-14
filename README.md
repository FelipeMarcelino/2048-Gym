# 2048-Gym

![Agent playing](https://github.com/FelipeMarcelino/2048-Gym/blob/master/model/img/play.gif)

This repository is a project about using DQN(Q-Learning) to play the Game 2048 and accelarate and accelerate the environment using [Numba](https://github.com/numba/numba)).  The algorithm used is from [Stable Baselines](https://github.com/hill-a/stable-baselines), and the environment is a custom [Open AI](https://github.com/openai) env.  The environment contains two types of representation for the board: binary and no binary. The first one uses a power two matrix to represent each tile of the board. On the contrary, no binary uses a raw matrix board. 

The model uses two different types of neural networks: CNN(Convolutional Neural Network), MLP(Multi-Layer Perceptron).
The agent performed better using CNN as an extractor for features than MLP. Probably it is because **CNN** can extract spatial features. As a result, the agent achieve a 2048 tile in 10% of the 1000 played games.

## Optuna

*Optuna* is an automatic hyperparameter optimization software framework, particularly designed
for machine learning. It features an imperative, *define-by-run* style user API. Thanks to our
*define-by-run* API, the code written with Optuna enjoys high modularity, and the user of
Optuna can dynamically construct the search spaces for the hyperparameters. 

There is a guide of how to use this library [here](https://github.com/optuna/optuna).

## Numba 

[Numba](https://github.com/numba/numba) is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.

There is a guide of how to use this library [here](https://github.com/numba/numba).



## Instalation 

* [OpenAI Gym](https://github.com/openai/gym)
* [TensorFlow 1.15.x](https://www.tensorflow.org) (2.x does not work)
* [Stable Baselines](https://github.com/hill-a/stable-baselines)
* [Optuna](https://github.com/optuna/optuna)
* [2048-Gym](https://github.com/FelipeMarcelino/2048-gym)
* [Numba](https://github.com/numba/numba)

Installing dependecies
`pip install -r [requirements_cpu.txt|requirements-gpu.txt]`,
choosing the appropriate file depending on whether you wish to run the models on a CPU or a GPU.

**OR**

Using conda environment

```
conda env create -f [conda_env_gpu.yml|conda_env_cpu.yml]
```

To install the environment, execute the following commands:
```sh
git clone https://github.com/FelipeMarcelino/2048-Gym/
cd 2048-gym/gym-game2048/
pip install -e .
``` 

## Running

```sh
usage: model_optimize.py [-h] --agent AGENT
                         [--tensorboard-log TENSORBOARD_LOG]
                         [--study-name STUDY_NAME] [--trials TRIALS]
                         [--n-timesteps N_STEPS] [--save-freq SAVE_FREQ]
                         [--save-dir SAVE_DIR] [--log-interval LOG_INTERVAL]
                         [--no-binary] [--seed SEED]
                         [--eval-episodes EVAL_EPISODES]
                         [--extractor EXTRACTOR] [--layer-normalization]
                         [--num-cpus NUM_CPUS] [--layers LAYERS [LAYERS ...]]
                         [--penalty PENALTY] [--load_path LOAD_PATH]
                         [--num_timesteps_log NUM_TIMESTEPS_LOG]

optional arguments:
  -h, --help            show this help message and exit
  --agent AGENT, -ag AGENT
                        Algorithm to use to train the model - DQN, ACER, PPO2
  --tensorboard-log TENSORBOARD_LOG, -tl TENSORBOARD_LOG
                        Tensorboard log directory
  --study-name STUDY_NAME, -sn STUDY_NAME
                        The name of study used for optuna to create the
                        database.
  --trials TRIALS, -tr TRIALS
                        The number of trials tested for optuna optimize. - 0
                        is the default setting and try until the script is
                        finish
  --n-timesteps N_STEPS, -nt N_STEPS
                        Number of timestems the model going to run.
  --save-freq SAVE_FREQ, -sf SAVE_FREQ
                        The interval between model saves.
  --save-dir SAVE_DIR, -sd SAVE_DIR
                        Save dictory models
  --log-interval LOG_INTERVAL, -li LOG_INTERVAL
                        Log interval
  --no-binary, -bi      Do not use binary observation space
  --seed SEED           Seed
  --eval-episodes EVAL_EPISODES, -ee EVAL_EPISODES
                        The number of episodes to test after training the
                        model
  --extractor EXTRACTOR, -ex EXTRACTOR
                        The extractor used to create the features from
                        observation space - (mlp or cnn)
  --layer-normalization, -ln
                        Use layer normalization - Only for DQN
  --num-cpus NUM_CPUS, -nc NUM_CPUS
                        Number of cpus to use. DQN only accept 1
  --layers LAYERS [LAYERS ...], -l LAYERS [LAYERS ...]
                        List of neurons to use in DQN algorithm. The number of
                        elements inside list going to be the number of layers.
  --penalty PENALTY, -pe PENALTY
                        How much penalize the model when choose a invalid
                        action
  --load_path LOAD_PATH, -lp LOAD_PATH
                        Load model from
  --num_timesteps_log NUM_TIMESTEPS_LOG, -ntl NUM_TIMESTEPS_LOG
                        Continuing timesteps for tensorboard_log
```

## Playing 

Play the game using trained agent.

```
python play_game.py 
```

**OBS: It is necessary to change the model path and agent inside play_game.py**

## Visualization

See best model actions using Tkinter.

```
python show_played_game.py
```
**OBS: It is necessary to change the pickle game data inside show_played_game.py**

