import numpy as np
import numba
from game_2048 import Game2048
from tqdm import tqdm

game = Game2048(4)

game.make_move(0)

for i in tqdm(range(10000000)):
    move = np.random.randint(4)
    game.make_move(move)

