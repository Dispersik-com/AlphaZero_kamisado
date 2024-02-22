import numpy as np
from .pieces import Monk


def kth_diagonals_indices(cell):
    right_diagonal = []
    left_diagonal = []

    i, j = cell

    # Check diagonal

    # diagonal down-right
    right_diagonal.extend([(i + x, j + x) for x in range(1, 8) if 0 <= i + x < 8 and 0 <= j + x < 8])
    # diagonal up-right
    right_diagonal.extend([(i - x, j + x) for x in range(1, 8) if 0 <= i - x < 8 and 0 <= j + x < 8])
    # diagonal down-left
    left_diagonal.extend([(i + x, j - x) for x in range(1, 8) if 0 <= i + x < 8 and 0 <= j - x < 8])
    # diagonal up-left
    left_diagonal.extend([(i - x, j - x) for x in range(1, 8) if 0 <= i - x < 8 and 0 <= j - x < 8])

    # Filter out moves outside the board
    right_diagonal = [(x, y) for (x, y) in right_diagonal if 0 <= x < 8 and 0 <= y < 8]
    left_diagonal = [(x, y) for (x, y) in left_diagonal if 0 <= x < 8 and 0 <= y < 8]

    return right_diagonal, left_diagonal


class KamisadoMoveValidator:

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