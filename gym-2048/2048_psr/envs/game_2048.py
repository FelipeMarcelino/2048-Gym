import numpy as np

class Game2048():

    def __init__(self, board_size: int) -> None:

        self.__board_size: int = board_size
        self.__total_score: int = 0
        self.__score: int = 0
        self.__temp_board = None
        self.__create_board()
        self.__add_two_or_four()
        self.__add_two_or_four()

    def __create_board(self) -> None:
        """Create the board game."""

        self.__board = np.zeros((self.__board_size, self.__board_size), dtype=np.uint32)



    def __add_two_or_four(self) -> None:
        """Add tile with number two."""

        # Coordinates to add a tile with number two
        line: int  = np.random.randint(0,  self.__board_size)
        column: int  = np.random.randint(0,  self.__board_size)
        while(self.__board[line][column] != 0):
            line = np.random.randint(0, self.__board_size)
            column = np.random.randint(0, self.__board_size)

        if np.random.uniform(0,1,1) >= 0.9:
            self.__board[line][column] = 4
        else:
            self.__board[line][column] = 2


    def __transpose(self, board):
        """Transpose a matrix."""

        temp = np.zeros((self.__board_size,self.__board_size), dtype=np.uint32)

        for line in range(self.__board_size):
            for column in range(self.__board_size):
                temp[column][line] = board[line][column]

        return temp

    def __reverse(self, board):
        """Reverse a matrix."""

        temp = np.zeros((self.__board_size,self.__board_size), dtype=np.uint32)

        for line in range(self.__board_size):
            for column in range(self.__board_size):
                temp[line][column] = board[self.__board_size - line - 1][column]

        return temp


    def __cover_up(self, board):
        """Cover the most antecedent zeros with non-zero number. """

        temp = np.zeros((self.__board_size,self.__board_size), dtype=np.uint32)

        for column in range(self.__board_size):
            up = 0
            for line in range(self.__board_size):
                if board[line][column] != 0 :
                    temp[up][column] = board[line][column]
                    up = up + 1

        return temp


    def __merge(self, board) -> np.array:
        """Verify if a merge is possible and execute."""

        for line in range(1, self.__board_size):
            for column in range(self.__board_size):
                if(board[line][column] == board[line - 1][column]):
                    self.__score = self.__score + (board[line][column] * 2)
                    board[line - 1][column] = board[line - 1][column] * 2
                    board[line][column] = 0
                else:
                    continue

        return board

    def __up(self) -> None:

        temp = self.__cover_up(self.__board)
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        self.__temp_board = temp

    def __down(self) -> None:

        temp = self.__reverse(self.__board)
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        temp = self.__reverse(temp)
        self.__temp_board = temp

    def __right(self) -> None:

        temp = self.__reverse(self.__transpose(self.__board))
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        temp = self.__transpose(self.__reverse(temp))
        self.__temp_board = temp

    def __left(self) -> None:

        temp = self.__transpose(self.__board)
        temp = self.__merge(temp)
        temp = self.__cover_up(temp)
        temp = self.__transpose(temp)
        self.__temp_board = temp


    def get_move_score(self) -> int:
        """Get the last score move."""

        return self.__score

    def get_total_score(self) -> int:
        """Get the total score gained until now."""

        return self.__total_score

    def set_board(self, board):
        """This function is only for test purpose."""
        self.__board = board

    def get_board(self):
        """Get the actual board."""

        return  self.__board

    def confirm_move(self):
        """Execute movement.""" 
        self.__board = self.__temp_board.copy()
        self.__total_score = self.__total_score + self.__score

    def make_move(self, move) -> None:
        """Make a move."""
        self.__score = 0

        if move == 0:
            self.__up()
        if move == 1:
            self.__down()
        if move == 2:
            self.__right()
        if move == 3:
            self.__left()
