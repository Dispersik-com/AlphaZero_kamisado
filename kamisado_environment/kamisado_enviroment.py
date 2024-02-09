import copy
import random
from .kamisado_board import KamisadoBoard
from .kamisado_move_validator import KamisadoMoveValidator


class KamisadoGame(KamisadoBoard, KamisadoMoveValidator):

    def __init__(self):
        # Initializing the game board with 0 (empty spaces)
        super().__init__()
        self.current_player = "White"
        self.last_move = None
        self.winner = None
        self.history_of_moves = []

    def __hash__(self):
        # Hashing the state based on significant attributes
        return hash(str(self.game_board))

    def switch_player(self):
        # Switching the current player for the next turn
        self.current_player = "Black" if self.current_player == "White" else "White"

    def make_move(self, start_cell, end_cell):

        # Checking if the move is valid
        if not self.is_valid_move(start_cell, end_cell):
            print("Invalid move!")
            return False

        start_row, start_col = start_cell
        end_row, end_col = end_cell

        # Moving the monk to the new position
        self.game_board[end_row][end_col] = self.game_board[start_row][start_col]
        self.game_board[start_row][start_col] = 0

        # init tuple of last move
        color_of_last_cell = self.color_dict[self.color_board[end_row][end_col]]
        self.last_move = ((end_row, end_col), color_of_last_cell)

        # write history of moves
        self.history_of_moves.append({"last move": self.last_move,
                                  "start cell": start_cell,
                                  "end cell": end_cell,
                                  "piece": str(self.game_board[end_row][end_col]),
                                  "player": self.current_player
                                      })
        # Switching the current player for the next turn
        self.switch_player()

    def find_monk(self, color_monk):
        # Iterate through the game board to find the coordinates of a monk with the specified colors
        for i, row in enumerate(self.game_board):
            for j, monk in enumerate(row):
                if isinstance(monk, int):
                    continue
                if monk.command_color == self.current_player and monk.self_color == color_monk:
                    return i, j

    def copy_and_apply_action(self, action):
        end_cell = action

        if self.last_move is None:
            self.set_first_piece()

        start_cell = self.find_monk(self.last_move[1])

        # Create a deep copy of the current state
        new_state = copy.deepcopy(self)
        # Perform the action in the copied state to create a new state
        new_state.make_move(start_cell, end_cell)

        return new_state

    def set_first_piece(self, color=None):
        """
        Set the piece that should move first.

        Args:
            color (str): Optional. The color of the piece that should move first.
                         If not provided, a random color is chosen from available colors.
        """
        if color is None:
            color = random.choice(list(self.color_dict.values()))
        cell = self.find_monk(color)
        self.last_move = (cell, color)

    def pass_move(self):
        """
        Pass the move to the next player.

        Returns:
            bool: True if the move is successfully passed, False otherwise.

        # According to Kamisado rules, if both players have no moves,
        # then the player who created the stalemate loses
        """
        color = self.last_move[1]
        row, col = self.find_monk(color)
        new_color = self.color_dict[self.color_board[row][col]]
        self.last_move = ((row, col), new_color)
        self.switch_player()

        if not self.get_legal_moves():
            self.winner = self.current_player
            return False
        return True

    def check_winner(self):
        # Check if any player has reached the opposite end and declare the winner
        for i in range(8):
            if self.game_board[7][i] and self.game_board[7][i].command_color == "Black":
                self.winner = "Black"
            if self.game_board[0][i] and self.game_board[0][i].command_color == "White":
                self.winner = "White"
        return None
