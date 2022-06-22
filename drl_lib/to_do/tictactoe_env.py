
from os import access
import numpy as np


class TicTacToeEnv():
    def __init__(self,size) -> None:
        self.size=size
        self.board = np.zeros((size, size))
        self.actions=np.arange(0,size*size)
        self.players = np.array([1,2])

    
    def state_id(self):
        state = 0
        for i in range(self.size):
            for j in range(self.size):
                state += self.board[i][j] * pow(self.size, i * self.size + j)
        return state

    def is_game_over(self) -> bool:
        if len(self.available_actions_ids())==0:
            return True
        # Check diagonals
        if self.board[0][0]==self.board[1][1] and self.board[1][1] == self.board[2][2] and self.board[0][0] !=0:
            return True
        if self.board[2][0]==self.board[1][1] and self.board[1][1] == self.board[0][2] and self.board[2][0] !=0:
            return True
        for i in self.size:
            # Check horizontals
            if self.board[i][0]==self.board[i][1] and self.board[i][1] == self.board[i][2] and self.board[i][0] !=0:
                return True
            # Check verticals
            elif self.board[0][i]==self.board[1][i] and self.board[1][i] == self.board[2][i] and self.board[0][i] !=0:
                return True
        return False

    def act_with_action_id(self, player:int,action_id: int):
        i= action_id // self.size
        j = action_id % self.size
        self.board[i][j]=player

    def score(self) -> float:
        return self.data.Score()

    def available_actions_ids(self):
        positions = []
        cpt =0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    positions.append(cpt)  
                    cpt+=1
        return np.array(positions)

        # Board : board de Tictactoe, taille 3x3
# 0 => Pas de pion
# 1 => Pion Joueur 1
# 2 => Pion Joueur 2
# 120
# 010
# 201
    def convertStateToBoard(self,state, b=3):
        if state == 0:
            return  np.zeros((self.size, self.size))
        digits = []
        while state:
            digits.append(int(state % b))
            state //= b
        digits = np.array(digits)
        return digits.reshape(self.size, self.size)