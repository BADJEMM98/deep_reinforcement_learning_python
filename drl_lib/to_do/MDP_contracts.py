from ..do_not_touch.contracts import MDPEnv
import numpy as np
class MyMDPEnv(MDPEnv):
    def __init__(self,states, rewards, actions,terminal_states, transition_matrix) -> None:
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.terminal_states = terminal_states
        self.transition_matrix = transition_matrix

    def states(self) -> np.ndarray:
        return np.asarray(self.states)

    def actions(self) -> np.ndarray:
        return np.asarray(self.actions)

    def rewards(self) -> np.ndarray:
        return np.asarray(self.rewards)

    def is_state_terminal(self, s: int) -> bool:
        return s in self.terminal_states

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:

        return self.transition_matrix[s,a,s_p,r]

    # def view_state(self, s: int):
    #     print("It's secret !")