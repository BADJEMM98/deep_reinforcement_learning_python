
import numpy as np


class TicTacToeEnv():
    def __init__(self,size) -> None:
        self.size=size
        self.board = np.zeros((size, size))
        self.actions=np.arange(0,size*size)
    
    def state_id(self) -> int:
        return self.data.StateId()

    def is_game_over(self) -> bool:
        return self.data.IsGameOver()

    def act_with_action_id(self, action_id: int):
        pass
    def score(self) -> float:
        return self.data.Score()

    def available_actions_ids(self) -> np.ndarray:
        return self.data.AvailableActionsIdsAsNumpy()

    def available_positions(self):
        positions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions