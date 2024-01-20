import numpy as np

class Monk:
    def __init__(self, command_color, self_color):
        self.command_color = command_color
        self.self_color = self_color

    def __str__(self):
        return f"{self.command_color[0]}-{self.self_color}"


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
        # Initializing the game board with None (empty spaces)
        self.game_board = np.full((8, 8), None)
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

    def make_move(self, start_cell, end_cell):

        start_row, start_col, end_row, end_col = start_cell, end_cell

        # Checking if the move is valid
        if not self.is_valid_move(start_row, start_col, end_row, end_col):
            print("Invalid move!")
            return

        # Moving the monk to the new position
        self.game_board[end_row][end_col] = self.game_board[start_row][start_col]
        self.game_board[start_row][start_col] = None

        # init tuple of last move
        self.last_move = ((end_row, end_col), self.color_board[end_row][end_col])

        # Switching the current player for the next turn
        self.switch_player()

    def _get_diagonals_by_cell(self, row, col):

        matrix = self.game_board

        diagonal_indices, diagonal_objects = None, None
        offset_by_object = matrix.shape[1] - 1 - col - row

        if self.current_player == "Black":

          main_diagonal = matrix.diagonal(col - row)[col+1:]
          secound_diagonal = matrix[:, ::-1].diagonal(offset_by_object)[offset_by_object+1:]

          diagonal_objects = main_diagonal, secound_diagonal

          main_indices = [(row+i+1, col+i+1)for i in range(len(main_diagonal))]
          secound_indices = [(row+i, col-i-1)for i in range(len(secound_diagonal))]

          diagonal_indices = main_indices, secound_indices

        else:

          main_diagonal = matrix.diagonal(col - row)[:col]
          secound_diagonal = matrix[:, ::-1].diagonal(offset_by_object)[:offset_by_object]

          diagonal_objects = main_diagonal, secound_diagonal

          main_indices = [(row-i-1, col-i-1)for i in range(len(main_diagonal))]
          secound_indices = [(row-i, col+i+1)for i in range(len(secound_diagonal))]

          diagonal_indices = main_indices, secound_indices

        return diagonal_indices, diagonal_objects

    def _get_sright_line_by_cell(self, row, col):
        
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
          giagonals = self._get_diagonals_by_cell(cell)
          stright_line = self._get_stright_by_cell(cell)
          return giagonals, stright_line
        else:
          indices = np.argwhere(self.game_board[1:])
          coordinates = [(i[0] + 1, i[1]) for i in indices]
          return coordinates

    def is_valid_move(self, start_row, start_col, end_row, end_col):
        # Check if the move is within the boundaries of the board
        if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
            return False

        # Get the monk at the starting position
        monk = self.game_board[start_row][start_col]

        # Check if there is a monk at the starting position and if it belongs to the current player
        if monk is None or monk.command_color != self.current_player:
            return False

        # Check if the destination position is empty
        if not (0 <= end_row < 8 and 0 <= end_col < 8 and self.game_board[end_row][end_col] is None):
            return False

        # Check for the correct direction of movement based on the current player
        if self.current_player == "White" and end_row >= start_row:
            return False
        elif self.current_player == "Black" and end_row <= start_row:
            return False

        # Check for movement on cells of the same color
        if not self.is_valid_color_move(start_row, start_col, end_row, end_col):
            return False

        return True

    def is_valid_color_move(self, start_row, start_col, end_row, end_col):
        if self.last_move is None:
          return True

        monk_color = self.game_board[start_row][start_row].self_color
        last_cell_color = self.color_dict[self.last_move[1]]
        # Check if the monk is moving on cells of the same color
        return monk_color == last_cell_color

    def switch_player(self):
        # Switching the current player for the next turn
        self.current_player = "Black" if self.current_player == "White" else "White"

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


    # def play_game(self):
    #     while True:
    #         self.print_board()

    #         start_row = int(input(f"{self.current_player}'s turn. Enter start row (0-7): "))
    #         start_col = int(input("Enter start column (0-7): "))
    #         end_row = int(input("Enter end row (0-7): "))
    #         end_col = int(input("Enter end column (0-7): "))

    #         self.make_move(start_row, start_col, end_row, end_col)

    #         winner = self.check_winner()
    #         if winner:
    #             print(f"{winner} player wins!")
    #             break
