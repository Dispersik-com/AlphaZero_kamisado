import numpy as np
from .pieces import Monk


class KamisadoBoard:
    # Initializing the colored board
    color_board = np.array([
        [7, 6, 5, 4, 3, 2, 1, 0],  # O A V P Y R G B
        [2, 7, 4, 1, 6, 3, 0, 5],  # R O P G A Y B V
        [1, 4, 7, 2, 5, 0, 3, 6],  # G P O R V B Y A
        [4, 5, 6, 7, 0, 1, 2, 3],  # P V A O B G R Y
        [3, 2, 1, 0, 7, 6, 5, 4],  # Y R G B O A V P
        [6, 3, 0, 5, 2, 7, 4, 1],  # A Y B V R O P G
        [5, 0, 3, 6, 1, 4, 7, 2],  # V B Y A G P O R
        [0, 1, 2, 3, 4, 5, 6, 7]  # B G R Y P V A O
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
        7: "O"  # Orange
    }

    def __init__(self):
        # Initializing the game board with 0 (empty spaces)
        self.game_board = np.zeros((8, 8), dtype=Monk)
        self.init_monk()

    def init_monk(self):
        # Initializing the initial distribution of monks
        self.game_board[0] = [Monk("Black", self.color_dict[7 - i]) for i in range(8)]
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