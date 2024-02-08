import numpy as np
import copy


class Monk:
    def __init__(self, command_color: str, self_color: str):
        self.command_color = command_color
        self.self_color = self_color

    def __str__(self):
        return f"{self.command_color[0]}-{self.self_color}"

    def __repr__(self) -> str:
        return f"{self.command_color[0]}-{self.self_color}"


def kth_diagonals_indices(cell):
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

    def __hash__(self):
        # Hashing the state based on significant attributes
        return hash(str(self.game_board))

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
                    print(f"{monk} ", end="")
                else:
                    print(f"e-{cell_color} ", end="")
            print()

    def switch_player(self):
        # Switching the current player for the next turn
        self.current_player = "Black" if self.current_player == "White" else "White"

    def _get_diagonals_by_cell(self, cell):

        # Calculate the indices of the two diagonals using the provided function
        first_diag, second_diag = kth_diagonals_indices(cell)

        # Define a filtering function based on the current player
        if self.current_player == "Black":
            filter_func = lambda x: x[0] > cell[0]
        else:
            filter_func = lambda x: x[0] < cell[0]

        # Apply the filtering function to both diagonals
        first_diag, second_diag = list(filter(filter_func, first_diag)), list(filter(filter_func, second_diag))

        # Extract objects on the filtered diagonals
        object_on_first_diag = [self.game_board[i][j] for i, j in first_diag]
        object_on_second_diag = [self.game_board[i][j] for i, j in second_diag]

        # Return the filtered diagonals and the objects on them
        return (first_diag, second_diag), (object_on_first_diag, object_on_second_diag)

    def _get_sright_line_by_cell(self, cell):
        row, col = cell

        # Determine the straight line objects and indices based on the current player
        if self.current_player == "Black":
            # For Black player, get the objects and indices below the current cell
            stright_line_objects = self.game_board[:, col].T[row + 1:]
            steps = len(stright_line_objects)
            stright_line_indices = list(enumerate([col] * steps, row + 1))
        else:
            # For White player, get the objects and indices above the current cell
            stright_line_objects = self.game_board[:, col].T[:row][::-1]
            stright_line_indices = list(enumerate([col] * row))[:row][::-1]

        # Return the calculated straight line indices and objects
        return stright_line_indices, stright_line_objects

    def get_legal_moves(self):
        if self.last_move is not None:
            _, monk_color = self.last_move
            # Find the cell coordinates of the monk with the specified color
            cell = self.find_monk(monk_color)

            # Get diagonals and objects on diagonals from the current cell
            diagonals, object_on_diag = self._get_diagonals_by_cell(cell)
            # Get straight line indices and objects from the current cell
            line_indices, line_objects = self._get_sright_line_by_cell(cell)

            # Unpack values from diagonals and objects
            first_diag, second_diag = diagonals
            object_on_first_diag, object_on_second_diag = object_on_diag

            # Combine diagonals and straight line indices with their corresponding objects
            steps_and_objects = list(zip([first_diag, second_diag, line_indices],
                                        [object_on_first_diag, object_on_second_diag, line_objects]))

            legal_moves = []
            # Iterate over steps and objects, find the first blocked monk, and collect legal moves
            for list_of_steps in steps_and_objects:
                index_of_block_monk = [i if isinstance(monk, Monk) else None for i, monk in enumerate(list_of_steps[1])]
                first_monk_index = next((i for i in index_of_block_monk if i is not None), None)
                legal_moves += list_of_steps[0][:first_monk_index]

            return legal_moves

        else:
            # If there is no last move, return all cell indices except the initial rows (8:56)
            indices = np.ndenumerate(self.game_board)
            return list(map(lambda x: x[0], list(indices)))[8:56]

    def is_valid_color_move(self, start_row, start_col):
        if self.last_move is None:
          return True

        monk_color = self.game_board[start_row][start_col].self_color
        last_cell_color = self.last_move[1]

        # Check if the monk is moving on cells of the same color
        return monk_color == last_cell_color

    def is_valid_move(self, start_cell, end_cell):

        start_row, start_col = start_cell
        end_row, end_col = end_cell

        # Get the monk at the starting position
        monk = self.game_board[start_row][start_col]

        # Check if there is a monk at the starting position and if it belongs to the current player
        if monk == 0 or monk.command_color != self.current_player:
            print("Monk or color not True")
            return False

        legal_moves = self.get_legal_moves()

        if (end_row, end_col) not in legal_moves:
            print('not in legal moves')
            return False

        if not self.is_valid_color_move(start_row, start_col):
            print("invalid color of monk")
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
        color_of_last_cell = self.color_dict[self.color_board[end_row][end_col]]
        self.last_move = ((end_row, end_col), color_of_last_cell)

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
        start_cell, end_cell = action

        # Create a deep copy of the current state
        new_state = copy.deepcopy(self)
        
        # Perform the action in the copied state to create a new state
        new_state.make_move(start_cell, end_cell)
        
        return new_state

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
