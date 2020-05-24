import pickle
import time
import numpy as np
import sys
from tkinter import *


played_game = pickle.load(open("best_episode.pkl", "rb"))

SIZE = 1200
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}

CELL_COLOR_DICT = {
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
}

FONT = ("Fira Code", 90, "bold")


class GameGrid(Frame):
    def __init__(self, boards):
        """
        Implementing game grid.

        Show a Tkinter window with the game board.

        Parameters
        ----------
        boards : List
            List of boards of a played game.    
        """
        Frame.__init__(self)

        self.grid()
        self.master.title("2048")
        self.boards = boards

        self.matrix = self.boards[0]
        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells()

        self.wait_visibility()
        self.after(10, self.make_move())

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(
                    master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2
                )
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def make_move(self):

        for board in self.boards:
            time.sleep(0.03)
            self.matrix = board
            self.update_grid_cells()

        time.sleep(10)
        sys.exit()


def remove_repeated_boards(played_game):
    "Remove remove repeat boards from list. Some movements are invalid, so they don't change the board."

    boards = played_game["boards"]
    actions = played_game["actions"]

    no_repeated_boards = []
    no_repeated_actions = []

    last_board = None
    for i, board in enumerate(boards):

        if isinstance(board, np.ndarray):
            if np.array_equal(last_board, board) is True:
                pass
            else:
                no_repeated_actions.append(actions[i])
                no_repeated_boards.append(board)
                last_board = board

        else:
            last_board = board
            no_repeated_boards.append(board)
            no_repeated_actions.append(None)

    return no_repeated_boards, no_repeated_actions


played_game = pickle.load(open("./best_episode_without_zero.pkl", "rb"))

no_repeated_boards, no_repeated_actions = remove_repeated_boards(played_game)


root = Tk()
root.attributes("-type", "dialog")
gamegrid = GameGrid(no_repeated_boards)
root.mainloop()

