import random

from ..do_not_touch.contracts import MDPEnv
import numpy as np
class MyMDPEnv(MDPEnv):
    def __init__(self,env_data) -> None:
        self.env_data = env_data

    def states(self) -> np.ndarray:
        return np.asarray(self.env_data["states"])

    def actions(self) -> np.ndarray:
        return np.asarray(self.env_data["actions"])

    def rewards(self) -> np.ndarray:
        return np.asarray(self.env_data["rewards"])

    def is_state_terminal(self, s: int) -> bool:
        return s in self.env_data["terminal_states"]

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        ridx = np.where(self.env_data["rewards"] == r)[0][0]
        return self.env_data["transition_matrix"][s,a,s_p,ridx]
