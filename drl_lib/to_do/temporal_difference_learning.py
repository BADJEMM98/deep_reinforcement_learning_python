import itertools
from collections import defaultdict
from random import random, choice, choices
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
from ..to_do.tictactoe_env import TicTacToeEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction

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
    num_episodes = 50000
    max_steps = 100
    alpha = 0.05
    gamma = 0.95

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})
    pi = defaultdict(lambda: {a: random() for a in env.available_actions_ids()})

    def choose_action(env):
        s = env.state_id()
        if random() < epsilon:
            action = choice(env.available_actions_ids())
        else:
            action = max(Q[s], key=Q[s].get)
        return action

    reward = 0

    for i_episode in range(1, num_episodes + 1):
        if i_episode % (num_episodes / 5) == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))
        env.reset()
        pred_state = 0
        pred_action = 0
        state1 = env.state_id()
        action1 = choose_action(env)
        Q[state1]
        t = 0
        while t < max_steps:
            env.act_with_action_id(env.players[1].sign, action1)

            if not env.is_game_over():
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign, rand_action)

            if env.is_game_over():
                reward = env.score()
                prediction = Q[pred_state][pred_action]
                target = reward + gamma * Q[state2][action2]
                Q[state1][action1] = Q[state1][action1] + alpha * (target - prediction)
                break
            else:
                state2 = env.state_id()
                action2 = choose_action(env)
                reward = env.score()

            # Learning the Q-value

            prediction = Q[state1][action1]
            target = reward + gamma * Q[state2][action2]
            Q[state1][action1] = Q[state1][action1] + alpha * (target - prediction)
            pred_state = state1
            state1 = state2
            pred_action = action2
            action1 = action2

            if env.is_game_over():
                break

    return Q


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


"""
def qLearning(env, num_episodes, discount_factor = 1.0,
							alpha = 0.6, epsilon = 0.1):

	Q-Learning algorithm: Off-policy TD control.
	Finds the optimal greedy policy while improving
	following an epsilon-greedy policy
	env = TicTacToeEnv
	# Action value function
	# A nested dictionary that maps
	# state -> (action -> action-value).
    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    pi = defaultdict(lambda: {a:0.0 for a in actions})

	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths = np.zeros(num_episodes),
		episode_rewards = np.zeros(num_episodes))	
	
	# Create an epsilon greedy policy function
	# appropriately for environment action space
	pi[env.state_id]= epsilon_greedy_policy(actions, Q, epsilon, env.state_id())
	
	# For every episode
	for ith_episode in range(num_episodes):
		
		# Reset the environment and pick the first action
		env.reset()
		state = env.state_id()
		for t in itertools.count():
			
			# get probabilities of all actions from current state
            pis = [pi[state][a] for a in env.available_actions_ids()]

            if max(pis) == 0.0:
                action = choice(env.available_actions_ids())
            else:
                action = choices(env.available_actions_ids(), weights=pis)[0]
			# choose action according to
			# the probability distribution
		
			# take action and get reward, transit to next state
			env.act_with_action_id(env.players[1].sign,a)
            
            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)

			# Update statistics
			stats.episode_rewards[ith_episode] += env.score()
			stats.episode_lengths[ith_episode] = t
			
			# TD Update
            best_action = max(Q[env.state_id],key=Q[env.state_id].get)
            #pi[env.state_id][best_action] = 1.0

			td_target = env.score() + discount_factor * Q[env.state_id][best_action]
			td_delta = td_target - Q[state][action]
			Q[state][action] += alpha * td_delta

			# done is True if episode terminated
			if done:
				break
				
			state = next_state

	return pi_and_Q = PolicyAndActionValueFunction(pi, Q)
"""

def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy
    """
    env = TicTacToeEnv()
    num_episodes = 10000
    discount_factor = 1.0
    epsilon = 0.1
    alpha = 0.6
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).

    actions = env.available_actions_ids()
    Q = defaultdict(lambda: {a:0.0 for a in actions})
    pi = defaultdict(lambda: {a:0.0 for a in actions})

    # Keeps track of useful statistics
    """stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))	
    """
    # Create an epsilon greedy policy function
    # appropriately for environment action space
    pi[env.state_id()]= epsilon_greedy_policy(actions, Q, epsilon, env.state_id())

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        env.reset()
        state = env.state_id()
        for t in itertools.count():
            
            # get probabilities of all actions from current state
            pis = [pi[state][a] for a in env.available_actions_ids()]

            if max(pis) == 0.0:
                action = choice(env.available_actions_ids())
            else:
                action = choices(env.available_actions_ids(), weights=pis)[0]
            # choose action according to
            # the probability distribution

            # take action and get reward, transit to next state
            env.act_with_action_id(env.players[1].sign,action)
            
            if not env.is_game_over():
                # faire jouer player[0]
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)

            # Update statistics
            #stats.episode_rewards[ith_episode] += env.score()
            #stats.episode_lengths[ith_episode] = t
            
            # TD Update
            best_action = max(Q[env.state_id()],key=Q[env.state_id()].get)
            #pi[env.state_id][best_action] = 1.0
            pi[env.state_id()]= epsilon_greedy_policy(actions, Q, epsilon, env.state_id())

            td_target = env.score() + discount_factor * Q[env.state_id()][best_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if env.is_game_over:
                break
                
    pi_and_Q = PolicyAndActionValueFunction(pi, Q)
    return pi_and_Q


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
    num_episodes = 50000
    max_steps = 100
    alpha = 0.05
    gamma = 0.95

    episode_lengths = defaultdict(float)
    episode_rewards = defaultdict(float)

    Q = defaultdict(lambda: {a: 0.0 for a in env.available_actions_ids()})

    def choose_action(state, env, Q):
        if np.random.uniform(0, 1) < epsilon:
            action = choice(env.available_actions_ids())
        else:
            action = max(Q[state], key=Q[state].get)
        return action

    for i_episode in range(1, num_episodes + 1):
        if i_episode % (num_episodes/5) == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))
        env.reset()
        state1 = env.state_id()
        action1 = choose_action(state1, env, Q)
        Q[env.state_id()]
        t=0
        while t < max_steps:
            env.act_with_action_id(action1)
            state2 = env.state_id()
            reward = env.score()
            action2 = choose_action(state2, env, Q)
            Q[env.state_id()]

            # Learning the Q-value
            Q[state1][action1] += alpha * (reward + gamma * Q[state2][action2] - Q[state1][action1])

            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            state1 = state2
            action1 = action2
            t += 1

            if env.is_game_over():
                break

    return Q

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
    print(sarsa_on_tic_tac_toe_solo())
    # print(q_learning_on_tic_tac_toe_solo())
    # print(expected_sarsa_on_tic_tac_toe_solo())

    # print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())
