from collections import defaultdict
from random import random
from random import random, choice, choices
import numpy as np


from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2


def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    # TODO
    pass


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    # TODO
    pass


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()

    def epsilon_greedy_policy(Q, epsilon, state, actions):
        A = defaultdict(
            lambda: {
                a: 1 * epsilon / len(env.available_actions_ids())
                for a in actions
            }
        )
        best_action = max(Q[state], key=Q[state].get)

        A[state][best_action] += (1.0 - epsilon)
        return A[state]


    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    C = defaultdict(lambda: {a: 0.0 for a in actions})

    pi = defaultdict(lambda: {a: random() for a in actions})
    target_policy = pi
    epsilon = 0.1
    num_episodes = 10000

    for i_episode in range(1, num_episodes + 1):

        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]

        a0 = choices(env.available_actions_ids(), weights=pis)[0]
        # print(a0)
        # faire jouer player[1]
        env.act_with_action_id(a0)

        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while (not env.is_game_over()):
            s = env.state_id()
            # print("s", s)
            # print("pis", pi[s])
            pis = [pi[s][a] for a in env.available_actions_ids()]

            a = choices(env.available_actions_ids(), weights=pis)[0]

            # faire jouer player[1]
            env.act_with_action_id(a)


            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0.0
        W = 1.0
        discount = 0.999

        for t in range(len(s_p_history))[::-1]:
            state, action, reward = s_p_history[t], a_history[t], r_history[t]
            G = discount * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            target_policy[state] = {a: 0.0 for a in actions}
            best_action = max(Q[state], key=Q[state].get)
            target_policy[state][best_action] = 1.0

            if action != best_action:
                break

            W = W * (target_policy[state][action] / pi[state][action])

    return Q, target_policy


def demo():
    # print(monte_carlo_es_on_tic_tac_toe_solo())
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())
    env = Env2()

    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())
    # print(env.reset_random(), env.state_id())



