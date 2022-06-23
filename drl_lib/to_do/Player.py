from typing import Optional
import numpy as np
from to_do.tictactoe_env import TicTacToeEnv


class Player():
    def __init__(self,sign:int,type:str) -> None:
        self.sign = sign
        self.is_winner = False
        self.type = type # 'H':human player, 'R': Random player, 'A'

    def play(self,available_actions_ids,state_id=None,policy=None) -> Optional[int]:
        action_id=None
        if self.type == 'H':
            while True:
                action_id = input("Please enter your action id: ")
                if action_id in available_actions_ids():
                    break
        elif self.type == 'R':
            action_id = np.random.choice(available_actions_ids)
        else:
            if state_id is not None and policy is not None:
                action_id = max(policy[state_id],key=policy[state_id].get)
        return action_id
