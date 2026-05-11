import numpy as np
from game import ConnectFour


class ConnectFourEnv:
    def __init__(self):
        self.game = ConnectFour()

    def reset(self):
        self.game.reset()
        return self.get_state()

    def get_state(self):
        return self.game.board.flatten().astype(np.float32)

    def get_valid_actions(self):
        return self.game.get_valid_actions()

    def opponent_move(self):
        valid_actions = self.game.get_valid_actions()
        if valid_actions:
            action = np.random.choice(valid_actions)
            self.game.apply_action(action, "opponent")
            return action
        return None

    def step(self, action):
        if not self.game.is_valid_action(action):
            raise ValueError(f"Column {action} is full")

        # Agent move
        self.game.apply_action(action, "agent")

        # Check if agent wins
        if self.game.check_win("agent"):
            return self.get_state(), 1, True, {"winner": "agent"}

        # Check draw after agent move
        if self.game.is_draw():
            return self.get_state(), 0, True, {"winner": None}

        # Opponent move
        opp_action = self.opponent_move()

        # Check if opponent wins
        if self.game.check_win("opponent"):
            return self.get_state(), -1, True, {"winner": "opponent", "opponent_action": opp_action}

        # Check draw after opponent move
        if self.game.is_draw():
            return self.get_state(), 0, True, {"winner": None, "opponent_action": opp_action}

        return self.get_state(), 0, False, {"winner": None, "opponent_action": opp_action}

    def render(self):
        self.game.render()