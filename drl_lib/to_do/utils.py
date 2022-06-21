import math
from random import random
import numpy as np

from .MDP_contracts import MyMDPEnv

from ..do_not_touch.result_structures import ValueFunction


def policy_eval(env, pi, Vs, theta=0.0000001):
    while True:
        delta = 0
        for s in env.states():
            v = Vs[s]
            Vs[s] = 0.0
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in env.rewards():
                        total += env.transition_probability(s, a, s_p, r) * (r + 0.999 * Vs[s_p])
                total *= pi[s][a]
                Vs[s] += total
            delta = max(delta, np.abs(v - Vs[s]))

        if delta < theta:
            break
    return Vs

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

def policy_improvement(env, pi, Vs):
    stable = True
    for s in env.states():
        old_pi_s = pi[s].copy()
        best_a = -1
        best_a_score = -99999999999
        for a in env.actions():
            total = 0
            for s_p in env.states():
                for r in env.rewards():
                    total += env.transition_probability(s, a, s_p, r) * (r + 0.999 * Vs[s_p])
            if total > best_a_score:
                best_a = a
                best_a_score = total

        pi[s] = {a:0.0 for a in env.actions()}
        pi[s][best_a] = 1.0
        #pi[s, :] = 0.0
        #pi[s, best_a] = 1.0
        if np.any(pi[s] != old_pi_s):
            stable = False
    return stable, pi

def policy_iteration(env):
    nb_cells = len(env.states())
    terminal_states = []
    for s in env.states():
        if env.is_state_terminal(s):
            terminal_states.append(s)
    V = np.random.random((nb_cells,))
    Vs: ValueFunction = {s: V[s] for s in env.states()}
    for state in terminal_states:
        Vs[state] = 0.0

    #pi = np.random.random((nb_cells, (len(env.actions))))
    pi = {s:{a:random() for a in env.actions()} for s in env.states()}

    for s in env.states():
        pi[s] = {a:v/total for total in (sum(pi[s].values()),) for a, v in pi[s].items()}
    for state in terminal_states:
        pi[state] = {a: 0.0 for a in env.actions()}
    while True:
        Vs = policy_eval(env, pi, Vs)

        stable, pi = policy_improvement(env, pi, Vs)
        if stable:
            return pi, Vs

def value_iteration(grid_env:MyMDPEnv, v , pi, gamma, theta):

    while True:
        delta = 0
        for s in grid_env.states():

            oldV = v[s]
            newV = []
            for a in grid_env.actions():
                for s_p in grid_env.states():
                    for r in grid_env.rewards():
                        newV.append(grid_env.transition_probability(s, a, s_p, r) * (r + gamma * v[s_p]))
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            v[s] = newV[bestState]
            delta = max(delta, np.abs(oldV - v[s]))

        if delta < theta:
            break

    for s in grid_env.states():
        newValues = []
        actions = []

        for a in grid_env.actions():
            for s_p in grid_env.states():
                for r in grid_env.rewards():
                        newValues.append(grid_env.transition_probability(s, a, s_p, r) * (r + gamma * v[s_p]))
                        actions.append(a)

        newValues = np.array(newValues)
        bestActionIDX = np.where(newValues == newValues.max())[0]
        bestActions = actions[bestActionIDX[0]]
        for i in range(len(grid_env.actions())):
            pi[s][i] = 0
        pi[s][bestActions] = 1.0

    return pi, v

# Board : board de Tictactoe, taille 3x3
# 0 => Pas de pion
# 1 => Pion Joueur 1
# 2 => Pion Joueur 2
# 120
# 010
# 201

def convertBoardToState(board):
    state = 0
    for i in range(3):
        for j in range(3):
            state += board[i][j] * pow(3, i * 3 + j)
    return state

def convertStateToBoard(state, b=3):
    if state == 0:
        return np.array([[0,0,0],[0,0,0],[0,0,0]])
    digits = []
    while state:
        digits.append(int(state % b))
        state //= b
    digits = np.array(digits)
    return digits.reshape(3,3)

