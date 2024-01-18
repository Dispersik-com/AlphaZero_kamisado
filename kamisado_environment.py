class Monk:
    def __init__(self, command_color, self_color):
        self.command_color = command_color
        self.self_color = self_color

    def __str__(self):
        return f"{self.command_color[0]}-{self.self_color}"


class KamisadoEnvironment:

    # Initializing the colored board
    color_board = [
        [7, 6, 5, 4, 3, 2, 1, 0],  # O A V P Y R G B
        [2, 7, 4, 1, 6, 3, 0, 5],  # R O P G A Y B V
        [1, 4, 7, 2, 5, 0, 3, 6],  # G P O R V B Y A
        [4, 5, 6, 7, 0, 1, 2, 3],  # P V A O B G R Y
        [3, 2, 1, 0, 7, 6, 5, 4],  # Y R G B O A V P
        [6, 3, 0, 5, 2, 7, 4, 1],  # A Y B V R O P G
        [5, 0, 3, 6, 1, 4, 7, 2],  # V B Y A G P O R
        [0, 1, 2, 3, 4, 5, 6, 7]   # B G R Y P V A O
    ]

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
        self.game_board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = "White"

        self.init_monk()

    def init_monk(self):
        # Initializing the initial distribution of monks
        self.game_board[0] = [Monk("Black", self.color_dict[i]) for i in range(8)]
        self.game_board[7] = [Monk("White", self.color_dict[i]) for i in range(8)]

    def print_board(self):
        for i in range(8):
            for j in range(8):
                cell_color = self.color_dict[self.color_board[i][j]]
                if self.game_board[i][j]:
                    monk = self.game_board[i][j]
                    print(f"{monk}_{cell_color} ", end="")
                else:
                    print(f"E-{cell_color} ", end="")
            print()

    def make_move(self, start_row, start_col, end_row, end_col):
        if not self.is_valid_move(start_row, start_col, end_row, end_col):
            print("Invalid move!")
            return

        self.game_board[end_row][end_col] = self.game_board[start_row][start_col]
        self.game_board[start_row][start_col] = None

        # Checking for opponent block
        if 0 < end_row < 7 and self.game_board[end_row + 1][end_col]:
            self.game_board[end_row + 1][end_col] = None

        self.switch_player()

    def is_valid_move(self, start_row, start_col, end_row, end_col):
      if not (0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8):
          return False

      monk = self.game_board[start_row][start_col]
      if not monk or monk.command_color != self.current_player:
          return False

      if not (0 <= end_row < 8 and 0 <= end_col < 8 and self.game_board[end_row][end_col] is None):
          return False

      # Проверка на правильное направление движения
      if self.current_player == "White" and end_row >= start_row:
          return False
      elif self.current_player == "Black" and end_row <= start_row:
          return False

      # Проверка на перемещение по клеткам того же цвета
      if not self.is_valid_color_move(start_row, start_col, end_row, end_col):
          return False

      return True

    def is_valid_color_move(self, start_row, start_col, end_row, end_col):
        start_color = (start_row + start_col) % 2
        end_color = (end_row + end_col) % 2
        return start_color == end_color

    def switch_player(self):
        self.current_player = "Black" if self.current_player == "White" else "White"

    def check_winner(self):
        for i in range(8):
            if self.game_board[0][i] and self.game_board[0][i].command_color == "Black":
                print("Black player wins!")
                return "Black"
            if self.game_board[7][i] and self.game_board[7][i].command_color == "White":
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
