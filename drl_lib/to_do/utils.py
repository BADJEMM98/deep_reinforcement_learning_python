import numpy as np


def policy_eval(env, pi, Vs, theta=0.0000001):
    while True:
        delta = 0
        for s in env.states():
            v = Vs[s]
            Vs[s] = 0.0
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + 0.999 * Vs[s_p])
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