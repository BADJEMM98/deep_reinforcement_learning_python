from collections import defaultdict
from random import random, choice, choices

import numpy as np

from ..to_do.tictactoe_env import TicTacToeEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
import numpy as np

def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    # TODO
    env = TicTacToeEnv()
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    pi = defaultdict(lambda: {a:random() for a in actions})
    num_episodes = 100

    for i in range(num_episodes):
        print(i)
        # if i%(num_episodes/10)==0:
        #     print("%f" % i/num_episodes)
        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]
        a0 = choices(env.available_actions_ids(), weights=pis)[0]

        # faire jouer player[1]
        env.act_with_action_id(env.players[1].sign,a0)

        # faire jouer player[0]
        rand_action = env.players[0].play(env.available_actions_ids())
        env.act_with_action_id(env.players[0].sign,rand_action)
        
        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while not env.is_game_over():
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]
            a = choices(env.available_actions_ids(), weights=pis)[0]

            # faire jouer player[1]
            env.act_with_action_id(env.players[1].sign,a)
            

            # faire jouer player[0]
            rand_action = env.players[0].play(env.available_actions_ids())
            env.act_with_action_id(env.players[0].sign,rand_action)
            
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            returns_sum[(s_t,a_t)] += G
            returns_count[(s_t,a_t)] += 1.0
            Q[s_t][a_t] = returns_sum[(s_t,a_t)]/returns_count[(s_t,a_t)]
            pi[s_t]={a:0.0 for a in actions}
            best_action = max(Q[s_t],key=Q[s_t].get)
            pi[s_t][best_action] = 1.0

        pi_and_Q = PolicyAndActionValueFunction(pi,Q)

    return pi_and_Q



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
    # TODO
    env = Env2()
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    pi = defaultdict(lambda: {a:random() for a in actions})

    num_episodes = 10000
    for i in range(num_episodes):
        env.reset()
        s0 = env.state_id()
        a0 = choice(env.available_actions_ids())

        env.act_with_action_id(a0)
        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while not env.is_game_over():
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]
            a = choices(env.available_actions_ids(), weights=pis)[0]
            env.act_with_action_id(a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())
        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            returns_sum[(s_t,a_t)] += G
            returns_count[(s_t,a_t)] += 1.0

            Q[s_t][a_t] = returns_sum[(s_t,a_t)]/returns_count[(s_t,a_t)]
            pi[s_t]={a:0.0 for a in actions}
            best_action = max(Q[s_t],key=Q[s_t].get)
            pi[s_t][best_action] = 1.0
            
            pi_and_Q = PolicyAndActionValueFunction(pi,Q)
            # pi_and_Q.pi = pi
            # pi_and_Q.q = Q

        # return pi.items(), Q.items()
        return pi_and_Q



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

    # TODO
    pass


def demo():
    print("monte_carlo_es_on_tic_tac_toe")
    print(monte_carlo_es_on_tic_tac_toe_solo())
    # print("on_policy_first_visit_monte_carlo_control_on_tic_tac_toe")
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print("off_policy_first_visit_monte_carlo_control_on_tic_tac_toe")
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    # print("secret env")
    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())