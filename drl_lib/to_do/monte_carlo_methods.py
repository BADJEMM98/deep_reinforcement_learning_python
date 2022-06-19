from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
import numpy as np
def es(S, A, iter_count, max_step):
    # Initialize
    pi = np.random.random((len(S), len(A)))
    for s in S:
        pi[s] /= np.sum(pi[s])

    q = np.random.random((len(S), len(A)))

    Returns = [[[] for a in A] for s in S]
    # Initialize end
    # Loop
    for it in range(iter_count):
        s0 = np.random.choice(S)
        a0 = np.random.choice(A)
        s = s0
        a = a0

        s_p, r, terminal = step(s0, a0)
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]

        step_count = 1
        while terminal == False and step_count < max_step:
            s = s_p
            a = np.random.choice(A, p=pi[s])

            s_p, r, terminal = step(s, a)
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
            step_count += 1

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

            Returns[s_t][a_t].append(G)
            q[s_t, a_t] = np.mean(Returns[s_t][a_t])
            pi[s_t, :] = 0.0
            pi[s_t, np.argmax(q[s_t])] = 1.0

    return pi, q

# q, n, R to initialize on 0 for each state
# pi to initialize >0
def on_policy(q,n,R,pi):
    while True:
        # Code suivant la politique pi ?
        # for s, a in step("episode") ?
        pass

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
    # TODO
    pass


def demo():
    # print(monte_carlo_es_on_tic_tac_toe_solo())
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    # print(monte_carlo_es_on_secret_env2())
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())
