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
    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    pi = defaultdict(lambda: {a:random() for a in actions})
    num_episodes = 50000

    for i in range(num_episodes):
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
            av_actions = env.available_actions_ids()
            if max(pis) == 0.0:
                a = choice(av_actions)
            else:
                a = choices(av_actions, weights=pis)[0]


            # faire jouer player[1]
            env.act_with_action_id(env.players[1].sign,a)
            
            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)
            game_over = env.is_game_over()
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0
        discount = 0.999
        for t in reversed(range(len(s_history))):
            G = discount * G + r_history[t]
            s_t = s_p_history[t]
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
            for a in actions:
                pi[s_t][a]=0.0
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
    def epsilon_greedy_policy(actions, Q, epsilon, state):
        A = defaultdict(
            lambda: {
                a: 1 * epsilon / len(actions)
                for a in actions
            }
        )
        best_action = max(Q[state], key=Q[state].get)
        A[state][best_action] += (1.0 - epsilon)
        return A[state]

    epsilon = 0.1
    num_episodes = 10000

    env = TicTacToeEnv()

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    actions = env.available_actions_ids()

    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    pi = defaultdict(lambda: {a: 1 for a in actions})

    for i_episode in range(1, num_episodes + 1):
        # if i_episode % (num_episodes/5) == 0:
            # print("\rEpisode {}/{}.".format(i_episode, num_episodes))
            # print(pi[0])

        env.reset()
        pair_history = []
        s_history = []
        a_history = []
        s_p_history = []
        r_history = []
        while not env.is_game_over():
            state = env.state_id()
            pi[state] = epsilon_greedy_policy(actions, Q, epsilon, state)
            keys = []
            for i in pi[state].keys():
                keys.append(i)
            if len(keys) > 1:
                a = choices(keys, weights=pi[state])[0]
            else:
                a = choice(keys)

            env.act_with_action_id(env.players[1].sign, a)

            if not env.is_game_over():
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign, rand_action)

            game_over = env.is_game_over()
            r = env.score()

            s_history.append(state)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(r)
            pair_history.append(((state, a), r))

        G = 0
        for ((s, a), r) in pair_history:
            first_occurence_idx = next(
                i for i, (s_a, r) in enumerate(pair_history) if s_a == (s, a))
            G = sum([r for ((s, a), r) in pair_history[first_occurence_idx:]])

            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1.0
            Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

            pi_and_Q = PolicyAndActionValueFunction(pi, Q)
    return pi_and_Q
    
def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToeEnv()

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    C = defaultdict(lambda: {a:0.0 for a in actions})

    pi = defaultdict(lambda: {a:random() for a in actions})
    target_policy = pi
    num_episodes = 50000

    
    for i_episode in range(1, num_episodes+1):
                
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
        s_p_history= [env.state_id()]
        r_history= [env.score()]
       
        while(not env.is_game_over()):
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]
            av_actions = env.available_actions_ids()
            if max(pis) == 0.0:
                a = choice(av_actions)
            else:
                a = choices(av_actions, weights=pis)[0]

            # faire jouer player[1]
            env.act_with_action_id(env.players[1].sign,a)
            
            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)
            
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())
            
        G = 0.0
        W = 1.0
        discount=0.999
        
        for t in range(len(s_p_history))[::-1]:
            state, action, reward = s_p_history[t],a_history[t],r_history[t]
            G = discount*G + reward
            C[state][action] += W
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])
            target_policy[state]={a:0.0 for a in actions}
            best_action = max(Q[state],key=Q[state].get)
            target_policy[state][best_action] = 1.0

            if action != best_action:
                break
                
            W = W * (target_policy[state][action]/pi[state][action])
        
    return PolicyAndActionValueFunction(target_policy,Q)
  


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
    def epsilon_greedy_policy(env, Q, epsilon, state):
        A = defaultdict(
            lambda: {
                a: 1 * epsilon / len(env.available_actions_ids())
                for a in env.available_actions_ids()
            }
        )
        best_action = max(Q[state], key=Q[state].get)

        A[state][best_action] += (1.0 - epsilon)
        return A[state]

    epsilon = 0.1
    num_episodes = 10000

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})
    pi = defaultdict(lambda: {a : random for a in env.available_actions_ids()})

    for i_episode in range(1, num_episodes + 1):
        # if i_episode % 1 == 0:
        #     print("\rEpisode {}/{}.".format(i_episode, num_episodes))

        env.reset()
        pair_history = []
        s_history = []
        while not env.is_game_over():
            state = env.state_id()
            pi[state] = epsilon_greedy_policy(env, Q, epsilon, state)
            keys = []
            for i in pi[state].keys():
                keys.append(i)
            if len(keys) > 1:
                a = choices(keys, weights=pi[state])[0]
            else:
                a = choice(keys)

            env.act_with_action_id(a)
            r = env.score()
            pair_history.append(((state, a), r))
            s_history.append(state)
        G = 0
        for ((s, a), r) in pair_history:
            first_occurence_idx = next(
                i for i, (s_a, r) in enumerate(pair_history) if s_a == (s, a))
            G = sum([r for ((s, a), r) in pair_history[first_occurence_idx:]])

            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1.0
            Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

            pi_and_Q = PolicyAndActionValueFunction(pi, Q)

    return pi_and_Q



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
    #print("monte_carlo_es_on_tic_tac_toe")
    print(monte_carlo_es_on_tic_tac_toe_solo())
    #print("on_policy_first_visit_monte_carlo_control_on_tic_tac_toe")
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print("off_policy_first_visit_monte_carlo_control_on_tic_tac_toe")
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    # print("secret env")
    # print(monte_carlo_es_on_secret_env2())
    #print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    #print(off_policy_monte_carlo_control_on_secret_env2())