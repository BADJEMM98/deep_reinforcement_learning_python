from random import random
from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from .MDP_contracts import MyMDPEnv
import numpy as np


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy Dict[s:int, V(s):float]
    """
    nb_cells = 7
    states = np.arange(nb_cells)
    actions = np.array([0, 1])
    rewards = np.array([-1.0, 0.0, 1.0])
    transition_matrix = np.zeros((len(states), len(actions), len(states), len(rewards)))
    for s in states[1:-1]:
        if s == 1:
            transition_matrix[s, 0, s - 1, 0] = 1.0
        else:
            transition_matrix[s, 0, s - 1, 1] = 1.0

        if s == nb_cells - 2:
            transition_matrix[s, 1, s + 1, 2] = 1.0
        else:
            transition_matrix[s, 1, s + 1, 1] = 1.0

    terminal_states = [states[0],states[-1]]


    env = MyMDPEnv(states=states,rewards=rewards,actions=actions,terminal_states=terminal_states,transition_matrix=transition_matrix)


    # TODO
    theta = 0.0000001
    V = np.random.random((nb_cells,))
    Vs:ValueFunction = {s:V[s] for s in env.states }
    Vs[0] = 0.0
    Vs[nb_cells - 1] = 0.0

    pi = {s:{a:random() for a in env.actions} for s in env.states}
    for s in env.states:
        pi[s] = {a:v/total for total in (sum(pi[s].values()),) for a, v in pi[s].items()}
    pi[0] = {a:0.0 for a in env.actions}
    pi[nb_cells - 1] = {a:0.0 for a in env.actions}

    while True:
        delta = 0
        for s in env.states:
            v = Vs[s]
            Vs[s] = 0.0
            for a in env.actions:
                total = 0.0
                for s_p in env.states:
                    for r in range(len(env.rewards)):
                        total += env.transition_probability(s, a, s_p, r) * (env.rewards[r] + 0.999 * Vs[s_p])
                total *= pi[s][a]
                Vs[s] += total
            delta = max(delta, np.abs(v - Vs[s]))
        if delta < theta:
            break

    return Vs


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass

# Initialisation de transition_matrix
def init_grid_transition(grid_size, states, actions, rewards):
    transition_matrix = np.zeros((len(states), len(actions), len(states), len(rewards)))
    for s in states:
        # Les movements verticaux
        # Direction haut
        if (int(s / grid_size) == 1) & (s % grid_size == 4):
            transition_matrix[s, 0, s - grid_size, 0] = 1.0
        elif int(s / grid_size) > 0:
            transition_matrix[s, 0, s - grid_size, 1] = 1.0

        # Direction bas

        if (int(s / grid_size) == grid_size - 2) & (s % grid_size == 4):
            transition_matrix[s, 1, s + grid_size, 2] = 1.0
        elif int(s / grid_size) < 4:
            transition_matrix[s, 1, s + grid_size, 1] = 1.0

        # Les movements de horizontaux
        # Direction gauche
        if s % grid_size > 0:
            transition_matrix[s, 2, s - 1, 1] = 1.0

        # Direction droite
        if s == grid_size - 2:
            transition_matrix[s, 3, s + 1, 0] = 1.0
        elif s == grid_size * grid_size - 2:
            transition_matrix[s, 3, s + 1, 2] = 1.0
        elif s % grid_size < 4:
            transition_matrix[s, 3, s + 1, 1] = 1.0

    for s_p in states:
        for a in actions:
            for r in range(len(rewards)):
                transition_matrix[grid_size - 1, a, s_p, r] = 0.0
                transition_matrix[grid_size * grid_size - 1, a, s_p, r] = 0.0

    return transition_matrix

def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    grid_size=5
    nb_cells= grid_size*grid_size
    states = np.arange(nb_cells)
    actions = np.array([0, 1,2,3]) # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
    rewards = np.array([-1.0, 0.0, 1.0])
    transition_matrix = init_grid_transition(grid_size, states, actions, rewards)

    terminal_states = [states[grid_size-1],states[nb_cells-1]]


    env = MyMDPEnv(states=states,rewards=rewards,actions=actions,terminal_states=terminal_states,transition_matrix=transition_matrix)


    # TODO
    theta = 0.0000001
    V = np.random.random((nb_cells,))
    Vs:ValueFunction = {s:V[s] for s in env.states }
    Vs[grid_size-1] = 0.0
    Vs[nb_cells - 1] = 0.0

    pi = {s:{a:random() for a in env.actions} for s in env.states}
    for s in env.states:
        pi[s] = {a:v/total for total in (sum(pi[s].values()),) for a, v in pi[s].items()}
    pi[grid_size-1] = {a:0.0 for a in env.actions}
    pi[nb_cells - 1] = {a:0.0 for a in env.actions}

    while True:
        delta = 0
        for s in env.states:
            v = Vs[s]
            Vs[s] = 0.0
            for a in env.actions:
                total = 0.0
                for s_p in env.states:
                    for r in range(len(env.rewards)):
                        total += env.transition_probability(s, a, s_p, r) * (env.rewards[r] + 0.999 * Vs[s_p])
                total *= pi[s][a]
                Vs[s] += total
            delta = max(delta, np.abs(v - Vs[s]))
        if delta < theta:
            break

    return Vs


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    
    # TODO
    pass


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():
    print(policy_evaluation_on_line_world())
    # print(policy_iteration_on_line_world())
    # print(value_iteration_on_line_world())

    # print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
    # print(value_iteration_on_grid_world())

    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())
    # print(value_iteration_on_secret_env1())
