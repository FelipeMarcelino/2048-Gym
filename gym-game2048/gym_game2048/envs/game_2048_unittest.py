import unittest
import numpy as np
from game_2048 import Game2048

class Game2048Test(unittest.TestCase):

    def test_up_total(self):
        game = Game2048(4)
        board = np.array([[2,2,2,0],[8,0,0,0],[0,2,0,0],[0,2,0,0]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(0)
        game.confirm_move()
        total = game.get_move_score()
        self.assertEqual(total, 4)

    def test_up_board(self):
        game = Game2048(4)
        board = np.array([[2,2,2,0],[8,0,0,0],[0,2,0,0],[0,2,0,0]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(0)
        game.confirm_move()
        board = game.get_board()
        np.testing.assert_array_equal(board, np.array([[2,4,2,0],[8,2,0,0],[0,0,0,0],[0,0,0,0]]))

    def test_down_total(self):
        game = Game2048(4)
        board = np.array([[0,0,0,0],[2,0,0,4],[2,8,8,16],[4,16,8,2]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(1)
        game.confirm_move()
        total = game.get_move_score()
        self.assertEqual(total, 20)

    def test_down_board(self):
        game = Game2048(4)
        board = np.array([[0,0,0,0],[2,0,0,4],[2,8,8,16],[4,16,8,2]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(1)
        game.confirm_move()
        board = game.get_board()
        np.testing.assert_array_equal(board, np.array([[0,0,0,0],[0,0,0,4],[4,8,0,16],[4,16,16,2]]))

    def test_right_total(self):
        game = Game2048(4)
        board = np.array([[0,2,0,0],[16,2,0,0],[32,16,0,0],[4,2,4,4]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(2)
        game.confirm_move()
        total = game.get_move_score()
        self.assertEqual(total, 8)

    def test_right_board(self):
        game = Game2048(4)
        board = np.array([[0,2,0,0],[16,2,0,0],[32,16,0,0],[4,2,4,4]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(2)
        game.confirm_move()
        board = game.get_board()
        np.testing.assert_array_equal(board, np.array([[0,0,0,2],[0,0,16,2],[0,0,32,16],[0,4,2,8]]))

    def test_left_total(self):
        game = Game2048(4)
        board = np.array([[8,16,2,2],[2,4,8,0],[8,0,0,0],[2,2,0,0]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(3)
        game.confirm_move()
        total = game.get_move_score()
        self.assertEqual(total, 8)

    def test_left_board(self):
        game = Game2048(4)
        board = np.array([[8,16,2,2],[2,4,8,0],[8,0,0,0],[2,2,0,0]], dtype=np.uint32)
        game.set_board(board)
        game.make_move(3)
        game.confirm_move()
        board = game.get_board()
        np.testing.assert_array_equal(board, np.array([[8,16,4,0],[2,4,8,0],[8,0,0,0],[4,0,0,0]]))

    def test_verify_game_state_zero(self):
        game = Game2048(4)
        board = np.array([[8,16,2,2],[2,4,8,0],[8,0,0,0],[2,2,0,0]], dtype=np.uint32)
        game.set_board(board)
        done = game.verify_game_state()
        self.assertEqual(done, False)
        
    def test_verify_game_state_merge(self):
        game = Game2048(4)
        board = np.array([[2,4,2,2],[2,16,8,4],[4,128,64,8],[16,4,2,4]], dtype=np.uint32)
        game.set_board(board)
        done = game.verify_game_state()
        self.assertEqual(done, False)
        




if __name__ == "__main__":
    unittest.main()
