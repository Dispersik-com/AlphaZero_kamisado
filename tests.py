import unittest
from kamisado_environment import KamisadoEnvironment, Monk

class TestKamisadoEnvironment(unittest.TestCase):

    def setUp(self):
        self.game = KamisadoEnvironment()

    def test_make_move(self):

        # Create moves
        moves = []
        moves.append(((7, 7), (6, 7), "W-O"))
        moves.append(((0, 5), (5, 5), "B-R"))
        moves.append(((6, 7), (3, 7), "W-O"))

        # Check if a moves is made correctly
        for move in moves:
          start_cell, end_cell, monk_color = move
          self.game.make_move(start_cell, end_cell)

          self.assertEqual(self.game.game_board[start_cell[0]][start_cell[1]], 0)
          self.assertEqual(str(self.game.game_board[end_cell[0]][end_cell[1]]), monk_color)


    def test_incorrect_make_move(self):
        # Checking that the move is not correct is not performed
        start_cell = (7, 7)
        end_cell = (0, 7)

        # move on other monk
        self.game.make_move(start_cell, end_cell)
        self.assertNotEqual(self.game.game_board[start_cell[0]][start_cell[1]], 0)
        self.assertNotEqual(str(self.game.game_board[end_cell[0]][end_cell[1]]), 'W-O')

        # self move
        self.game.make_move(end_cell, end_cell)
        self.game.print_board()
        self.assertNotEqual(self.game.game_board[start_cell[0]][start_cell[1]], 0)
        self.assertEqual(str(self.game.game_board[start_cell[0]][start_cell[1]]), 'W-O')


    def test_get_legal_moves(self):
        # Check if legal moves are obtained correctly
        self.game.make_move((7,4), (5, 4))
        legal_moves = self.game.get_legal_moves()
        
        # correct moves
        for cell in [(2, 7), (6, 5), (5, 0)]:
          self.assertIn(cell, legal_moves)
        
        # next step
        self.game.make_move((0,5), (6, 5))
        legal_moves = self.game.get_legal_moves()
        # show_legal_moves(legal_moves)

        # correct moves with next step
        for cell in [(2, 7), (1, 4), (1, 0)]:
          self.assertIn(cell, legal_moves)

        # create monks who block moves
        self.game.game_board[2][7] = Monk("Black", "O")
        self.game.game_board[1][4] = Monk("Black", "O")
        self.game.game_board[1][0] = Monk("Black", "O")
        legal_moves = self.game.get_legal_moves()

        # correct moves with block monks
        for cell in [(2, 7), (1, 4), (1, 0)]:
          self.assertNotIn(cell, legal_moves)


    def test_check_winner(self):
        # Change the command color of the monk for checking winning

        self.game.game_board[7][7].command_color = "Black"
        self.assertEqual(self.game.check_winner(), "Black")
        # reset changes
        self.game.game_board[7][7].command_color = "White"

        self.game.game_board[0][7].command_color = "White"
        self.assertEqual(self.game.check_winner(), "White")
        # reset changes
        self.game.game_board[0][7].command_color = "Black"


if __name__ == '__main__':
    unittest.main()
