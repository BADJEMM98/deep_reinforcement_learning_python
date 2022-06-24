from os import access
import numpy as np

from ..to_do.Player import Player


class TicTacToeEnv():
    def __init__(self,size=3) -> None:
        self.size=size
        self.board = np.zeros((size, size))
        self.actions=np.arange(0,size*size)
        self.players = np.array([Player(1,'R'),Player(2,'A')])

    
    def state_id(self):
        state = 0
        for i in range(self.size):
            for j in range(self.size):
                state += self.board[i][j] * pow(self.size, i * self.size + j)
        return state

    def is_game_over(self) -> bool:
        if len(self.available_actions_ids())==0:
            return True
        else:
            # Check diagonals
            if self.board[0][0]==self.board[1][1] and self.board[1][1] == self.board[2][2] and self.board[0][0] !=0:
                if self.players[0].sign == self.board[0][0]:
                    self.players[0].is_winner = True
                else:
                    self.players[1].is_winner = True
                return True
            elif self.board[2][0]==self.board[1][1] and self.board[1][1] == self.board[0][2] and self.board[2][0] !=0:
                if self.players[0].sign == self.board[2][0]:
                    self.players[0].is_winner = True
                else:
                    self.players[1].is_winner = True
                return True
            else:
                for i in range(self.size):
                    # Check horizontals
                    if self.board[i][0]==self.board[i][1] and self.board[i][1] == self.board[i][2] and self.board[i][0] !=0:
                        if self.players[0].sign == self.board[i][0]:
                            self.players[0].is_winner = True
                        else:
                            self.players[1].is_winner = True
                        return True
                    # Check verticals
                    elif self.board[0][i]==self.board[1][i] and self.board[1][i] == self.board[2][i] and self.board[0][i] !=0:
                        if self.players[0].sign == self.board[0][0]:
                            self.players[0].is_winner = True
                        else:
                            self.players[1].is_winner = True
                        return True
        return False

    def act_with_action_id(self, player_sign:int,action_id: int):
        i= action_id // self.size
        j = action_id % self.size
        self.board[i][j]=player_sign

    def score(self) -> float:
        score = 0
        if self.players[1].is_winner:
            score = 10
        
        return score

    def available_actions_ids(self):
        positions = []
        cpt =0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    positions.append(cpt)  
                    cpt+=1
        return np.array(positions)

    def reset(self):
        self.board = np.zeros((self.size, self.size))

    def convertStateToBoard(self,state, b=3):
        if state == 0:
            return  np.zeros((self.size, self.size))
        digits = []
        while len(digits) < 9:
            digits.append(int(state % b))
            state //= b
        digits = np.array(digits)
        return digits.reshape(self.size, self.size)