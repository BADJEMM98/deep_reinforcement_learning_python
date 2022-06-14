from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from .MDP_contracts import MyMDPEnv
from random import random
import numpy as np


def policy_eval(env, pi, Vs, theta=0.0000001):
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
                total *= pi[s, a]
                Vs[s] += total
            delta = max(delta, np.abs(v - Vs[s]))

        if delta < theta:
            break
    return Vs


def policy_improvement(env, pi, Vs):
    stable = True
    for s in env.states():
        old_pi_s = pi[s].copy()
        best_a = -1
        best_a_score = -99999999999
        for a in env.actions():
            total = 0
            for s_p in env.states():
                for r in range(len(env.rewards())):
                    total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + 0.999 * Vs[s_p])
            if total > best_a_score:
                best_a = a
                best_a_score = total

        pi[s] = {a:0.0 for a in env.actions()}
        pi[s][best_a]=1.0

        if np.any(pi[s] != old_pi_s):
            stable = False
    return stable, pi



def q_value(env, v, s,a,gamma):
    q=0
    for (p_sPrime, sPrime, r_ss_a, done) in env.transition_probability:
        q += p_sPrime * (r_ss_a+gamma + v[sPrime])

    return q

# value_intération
def interate_values(grid_env, v , pi, gamma, theta):


    while True:
        DELTA = 0
        for s in grid_env.states():

            oldV = v[s]
            newV = []
            for a in grid_env.actions():
                for s_p in grid_env.states():
                    for r in range(len(grid_env.rewards())):

                        newV.append(grid_env.transition_probability(s, a, s_p, r) * (r + gamma * v[s_p]))
            newV = np.array(newV)
            bestV = np.where(newV == newV.max())[0]
            bestState = np.random.choice(bestV)
            v[s] = newV[bestState]
            DELTA = max(DELTA, np.abs(oldV - v[s]))

        if DELTA >= theta:
            break

    for s in grid_env.states():
        newValues = []
        actions = []

        for a in grid_env.actions():
            for s_p in grid_env.states():
                for r in range(len(grid_env.rewards())):
                        newValues.append(grid_env.transition_probability(s, a, s_p, r) * (r + gamma * v[s_p]))
                actions.append(a)
        newValues = np.array(newValues)
        bestActionIDX = np.where(newValues == newValues.max())[0]
        bestActions = actions[bestActionIDX[0]]
        pi[s][bestActions]= 1.0


    return v, pi

def v_itération(env):

    nb_cells = len(env.states())
    V = np.random.random((nb_cells,))
    Vs: ValueFunction = {s: V[s] for s in env.states()}
    Vs[0] = 0.0
    Vs[nb_cells - 1] = 0.0
    gamma = 0.99
    theta = 0.00001

    # pi = np.random.random((nb_cells, (len(env.actions))))
    pi = {s: {a: random() for a in env.actions()} for s in env.states()}

    for s in env.states():
        if s in env.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0.0

    pi[s] = {a: v / total for total in (sum(pi[s].values()),) for a, v in pi[s].items()}

    pi[0] = {a: 0.0 for a in env.actions()}
    pi[nb_cells - 1] = {a: 0.0 for a in env.actions()}

    converged = False
    i = 0


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

    pi = np.random.random((nb_cells, (len(env.actions))))

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
                total *= pi[s, a]
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
    nb_cells = 7
    states = np.arange(nb_cells)
    actions = np.array([0, 1])
    rewards = np.array([-1.0, 0.0, 1.0])
    transition_matrix = np.zeros((len(states), len(actions), len(states), len(rewards)))  # p
    for s in states[1:-1]:
        if s == 1:
            transition_matrix[s, 0, s - 1, 0] = 1.0
        else:
            transition_matrix[s, 0, s - 1, 1] = 1.0

        if s == nb_cells - 2:
            transition_matrix[s, 1, s + 1, 2] = 1.0
        else:
            transition_matrix[s, 1, s + 1, 1] = 1.0
    terminal_states = [states[0], states[-1]]

    env_data = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "terminal_states": terminal_states,
        "transition_matrix": transition_matrix
    }
    env = MyMDPEnv(env_data)

    return policy_iteration(env)

def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    pass


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
    grid_size = 5
    gamma = 1.0
    theta = 1e-10
    nb_cells = grid_size * grid_size
    states = np.arange(nb_cells)
    actions = np.array([0, 1, 2, 3])  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
    rewards = np.array([-1.0, 0.0, 1.0])
    transition_matrix = init_grid_transition(grid_size, states, actions, rewards)

    terminal_states = [states[grid_size - 1], states[nb_cells - 1]]

    env = MyMDPEnv(states=states, rewards=rewards, actions=actions, terminal_states=terminal_states,
                   transition_matrix=transition_matrix)

    # TODO
    theta = 0.0000001
    V = np.random.random((nb_cells,))
    Vs: ValueFunction = {s: V[s] for s in env.states}
    Vs[grid_size - 1] = 0.0
    Vs[nb_cells - 1] = 0.0

    pi = {s: {a: random() for a in env.actions} for s in env.states}
    for s in env.states:
        pi[s] = {a: v / total for total in (sum(pi[s].values()),) for a, v in pi[s].items()}
    pi[grid_size - 1] = {a: 0.0 for a in env.actions}
    pi[nb_cells - 1] = {a: 0.0 for a in env.actions}


    # TODO
    v, policy = interate_values(env, V, pi, gamma, theta)
    return  v, policy

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
    # print(policy_evaluation_on_line_world())
    # print(policy_iteration_on_line_world())
    #  print(value_iteration_on_line_world())

    # print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
     print(value_iteration_on_grid_world())

    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())
    # print(value_iteration_on_secret_env1())
