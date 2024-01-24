import numpy as np


class Monk:
    def __init__(self, command_color: str, self_color: str):
        self.command_color = command_color
        self.self_color = self_color

    def __str__(self):
        return f"{self.command_color[0]}-{self.self_color}"

    def __repr__(self) -> str:
        return f"{self.command_color[0]}-{self.self_color}"


def kth_diagonals_indices(matrix, cell):
    right_diagonal = []
    left_diagonal = []

    i, j = cell

    # Check diagonal
    right_diagonal.extend([(i + x, j + x) for x in range(1, 8) if 0 <= i + x < 8 and 0 <= j + x < 8])  # diagonal down-right
    right_diagonal.extend([(i - x, j + x) for x in range(1, 8) if 0 <= i - x < 8 and 0 <= j + x < 8])  # diagonal up-right
    left_diagonal.extend([(i + x, j - x) for x in range(1, 8) if 0 <= i + x < 8 and 0 <= j - x < 8])  # diagonal down-left
    left_diagonal.extend([(i - x, j - x) for x in range(1, 8) if 0 <= i - x < 8 and 0 <= j - x < 8])  # diagonal up-left

    # Filter out moves outside the board
    right_diagonal = [(x, y) for (x, y) in right_diagonal if 0 <= x < 8 and 0 <= y < 8]
    left_diagonal = [(x, y) for (x, y) in left_diagonal if 0 <= x < 8 and 0 <= y < 8]
    
    return right_diagonal, left_diagonal


class KamisadoEnvironment:

    # Initializing the colored board
    color_board = np.array([
        [7, 6, 5, 4, 3, 2, 1, 0],  # O A V P Y R G B
        [2, 7, 4, 1, 6, 3, 0, 5],  # R O P G A Y B V
        [1, 4, 7, 2, 5, 0, 3, 6],  # G P O R V B Y A
        [4, 5, 6, 7, 0, 1, 2, 3],  # P V A O B G R Y
        [3, 2, 1, 0, 7, 6, 5, 4],  # Y R G B O A V P
        [6, 3, 0, 5, 2, 7, 4, 1],  # A Y B V R O P G
        [5, 0, 3, 6, 1, 4, 7, 2],  # V B Y A G P O R
        [0, 1, 2, 3, 4, 5, 6, 7]   # B G R Y P V A O
    ])

    # Dictionary for visual representation of colors
    color_dict = {
        0: "B",  # Brown
        1: "G",  # Green
        2: "R",  # Red
        3: "Y",  # Yellow
        4: "P",  # Pink
        5: "V",  # Violet
        6: "A",  # Aqua
        7: "O"   # Orange
    }


    def __init__(self):
        # Initializing the game board with 0 (empty spaces)
        self.game_board = np.zeros((8, 8), dtype=Monk)
        self.current_player = "White"
        self.last_move = None
        # Initializing the monks on the board
        self.init_monk()


    def init_monk(self):
        # Initializing the initial distribution of monks
        self.game_board[0] = [Monk("Black", self.color_dict[7-i]) for i in range(8)]
        self.game_board[7] = [Monk("White", self.color_dict[i]) for i in range(8)]

    def print_board(self):
        # Printing the game board
        for i in range(8):
            for j in range(8):
                cell_color = self.color_dict[self.color_board[i][j]]
                if self.game_board[i][j]:
                    monk = self.game_board[i][j]
                    print(f"{monk}_{cell_color} ", end="")
                else:
                    print(f"E-{cell_color} ", end="")
            print()

    def switch_player(self):
        # Switching the current player for the next turn
        self.current_player = "Black" if self.current_player == "White" else "White"

    def _get_diagonals_by_cell(self, cell):
        row, col = cell
        matrix = self.game_board

        first_diag, second_diag = kth_diagonals_indices(matrix, cell)

        # object_on_first_diag = [self.game_board[i][j] for i, j in first_diag]
        # object_on_second_diag = [self.game_board[i][j] for i, j in second_diag]

        return first_diag, second_diag


    def _get_sright_line_by_cell(self, cell):
        row, col = cell

        if self.current_player == "Black":
          stright_line_objects = self.game_board[:,col].T[row:]
          steps = len(stright_line_objects)
          stright_line_indices = list(enumerate([col]*steps))[::-1][row+1:]
        else:
          stright_line_objects = self.game_board[:,col].T[:row+1]
          stright_line_indices = list(enumerate([col]*row))[::-1][:row+1]

        return stright_line_indices, stright_line_objects


    def get_legal_moves(self):
        if self.last_move is not None:
          cell, _ = self.last_move
          diagonals = self._get_diagonals_by_cell(cell)
          stright_line = self._get_sright_line_by_cell(cell)
          # TODO: add this function
          return 
        else:
          indices = np.ndenumerate(self.game_board)
          return list(map(lambda x: x[0], list(indices)))[8:56]


    def is_valid_color_move(self, start_row, start_col, end_row, end_col):
        if self.last_move is None:
          return True

        monk_color = self.game_board[start_row][start_row].self_color
        last_cell_color = self.color_dict[self.last_move[1]]
        # Check if the monk is moving on cells of the same color
        return monk_color == last_cell_color


    def is_valid_move(self, start_cell, end_cell):

        start_row, start_col = start_cell
        end_row, end_col = end_cell

        # Get the monk at the starting position
        monk = self.game_board[start_row][start_col]

        # Check if there is a monk at the starting position and if it belongs to the current player
        if monk == 0 or monk.command_color != self.current_player:
            return False

        giagonals = self.get_legal_moves()
        # legal_moves = giagonals[0][0] + stright_line[0][0]

        # print(legal_moves)

        # if (end_row, end_col) not in legal_moves:
        #     return False

        # Check for movement on cells of the same color
        if not self.is_valid_color_move(start_row, start_col, end_row, end_col):
            return False

        return True


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
        self.last_move = ((end_row, end_col), self.color_board[end_row][end_col])

        # Switching the current player for the next turn
        self.switch_player()


    def find_monk(self, command_color, self_color):
        for i, row in enumerate(self.game_board):
          for j, monk in enumerate(row):
              print(monk)
              if monk.command_color == command_color and monk.self_color == self_color:
                return i, j
        return None


    def check_winner(self):
        # Check if any player has reached the opposite end and declare the winner
        for i in range(8):
            if self.game_board[7][i] and self.game_board[7][i].command_color == "Black":
                print("Black player wins!")
                return "Black"
            if self.game_board[0][i] and self.game_board[0][i].command_color == "White":
                print("White player wins!")
                return "White"
        return None
