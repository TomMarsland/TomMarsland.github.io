import numpy as np


class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((6, 7), dtype=int)

    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        return self.board.copy()

    def copy(self):
        new_game = ConnectFour()
        new_game.board = np.copy(self.board)
        return new_game

    def get_valid_actions(self):
        return [i for i, x in enumerate(self.board[0]) if x == 0]

    def is_valid_action(self, col):
        return self.board[0][col] == 0

    def apply_action(self, col, player):
        if player == "agent":
            val = 1
        else:
            val = -1

        if not self.is_valid_action(col):
            raise ValueError(f"Column {col} is full")

        for i in range(self.rows - 1, -1, -1):
            if self.board[i][col] == 0:
                self.board[i][col] = val
                return i

    def check_win(self, player):
        if player == "agent":
            val = 1
        else:
            val = -1

        # Horizontal
        for i in range(self.rows):
            for j in range(self.cols - 3):
                if (
                    self.board[i][j] == val
                    and self.board[i][j + 1] == val
                    and self.board[i][j + 2] == val
                    and self.board[i][j + 3] == val
                ):
                    return True

        # Vertical
        for i in range(self.rows - 3):
            for j in range(self.cols):
                if (
                    self.board[i][j] == val
                    and self.board[i + 1][j] == val
                    and self.board[i + 2][j] == val
                    and self.board[i + 3][j] == val
                ):
                    return True

        # Diagonal: top-left to bottom-right
        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                if (
                    self.board[i][j] == val
                    and self.board[i + 1][j + 1] == val
                    and self.board[i + 2][j + 2] == val
                    and self.board[i + 3][j + 3] == val
                ):
                    return True

        # Diagonal: top-right to bottom-left
        for i in range(self.rows - 3):
            for j in range(3, self.cols):
                if (
                    self.board[i][j] == val
                    and self.board[i + 1][j - 1] == val
                    and self.board[i + 2][j - 2] == val
                    and self.board[i + 3][j - 3] == val
                ):
                    return True

        return False

    def is_draw(self):
        return np.all(self.board[0] != 0)

    def render(self):
        print(self.board)


C4 = ConnectFour()

print(C4.board[0][0])