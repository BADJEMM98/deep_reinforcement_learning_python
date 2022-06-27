from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3

from ..to_do.tictactoe_env import TicTacToeEnv
from collections import defaultdict
from random import random, choice, choices
import numpy as np

import matplotlib.pyplot as plt

def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv()

    epsilon = 0.9
    total_episodes = 10000
    max_steps = 100
    alpha = 0.05
    gamma = 0.95

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    pi = defaultdict(lambda: {a: random() for a in actions})

    def choose_action(state):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = choice(env.available_actions_ids())
        else:
            action = max(Q[state],key=Q[state].get)
        return action

    reward = 0

    for episode in range(1, total_episodes+1):
        env.reset()
        pred_state = 0
        pred_action = 0
        state1 = env.state_id()
        action1 = choose_action(state1)
        while not env.is_game_over():
            env.act_with_action_id(env.players[1].sign, action1)

            if not env.is_game_over():
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign, rand_action)

            state2 = env.state_id()
            reward = env.score()

            if env.is_game_over():
                state2 = state1
                state1 = pred_state
                action1 = pred_action
            else:
                action2 = choose_action(state2)

            # Learning the Q-value
            Q[state1][action1] = Q[state1][action1] + alpha * (reward + gamma * Q[state2][action2] - Q[state1][action1])
            pred_state = state1
            state1 = state2
            pred_action = action2
            action1 = action2

    return Q


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    epsilon = 0.9
    total_episodes = 10000
    alpha = 0.05
    gamma = 0.95

    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})

    def choose_action(state, env, Q):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = choice(env.available_actions_ids())
        else:
            action = max(Q[state], key=Q[state].get)
        return action

    test = []
    i = 1
    for episode in range(1, total_episodes + 1):

        env.reset()
        state1 = env.state_id()
        action1 = choose_action(state1, env, Q)
        Q[env.state_id()]
        t=0
        while not env.is_game_over():
            env.act_with_action_id(action1)
            state2 = env.state_id()
            reward = env.score()
            action2 = choose_action(state2, env, Q)
            Q[env.state_id()]

            # Learning the Q-value
            Q[state1][action1] += alpha * (reward + gamma * Q[state2][action2] - Q[state1][action1])

            episode_rewards[episode] += reward
            episode_lengths[episode] = t

            state1 = state2
            action1 = action2
            t+=1

    for i in range(total_episodes):
        print(episode_rewards[i], episode_lengths[i])
    return Q[0], test

def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def demo():
    # print(sarsa_on_tic_tac_toe_solo())
    # print(q_learning_on_tic_tac_toe_solo())
    # print(expected_sarsa_on_tic_tac_toe_solo())
    #
    print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())
