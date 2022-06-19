import operator
import random
import pygame
import numpy as np
from drl_lib.do_not_touch.mdp_env_wrapper import Env1
from drl_lib.to_do.MDP_contracts import MyMDPEnv
from drl_lib.to_do.lw_agent import LineWorldAgent
from drl_lib.to_do.dynamic_programming import policy_iteration_on_line_world

pygame.init()
WIDTH,HEIGHT=600,500
WIN = pygame.display.set_mode((HEIGHT,WIDTH))
pygame.display.set_caption("Line world")

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (124,252,0)
RED = (255,0,0)

block_size = 50
start_x=100
start_y = 50


clock = pygame.time.Clock()

def line_world(env,x_agent=None,y_agent=None):
    columns =len(env.states())
    for x in range(start_x,start_x+columns*block_size,block_size):
        s = int((x-start_x)/block_size)
        
        rect = pygame.Rect(x,start_y,block_size,block_size)
        if x == start_x:
            pygame.draw.rect(WIN,RED,rect,0)
        if x ==  start_x+(columns-1)*block_size:
            pygame.draw.rect(WIN,GREEN,rect,0)
        else:
            pygame.draw.rect(WIN,BLACK,rect,1)
    if x_agent is not None and y_agent is not None:
        pygame.draw.circle(WIN,BLACK,(x_agent,y_agent),10,0)
            
    

def draw_window(env,x_agent,y_agent):
    line_world(env,x_agent,y_agent)
    pygame.display.flip()

def main():
    run=True
    columns = 7
    states = np.arange(columns)
    actions = np.array([0, 1])
    rewards = np.array([-1.0, 0.0, 1.0])
    transition_matrix = np.zeros((len(states), len(actions), len(states), len(rewards)))
    for s in states[1:-1]:
        if s == 1:
            transition_matrix[s, 0, s - 1, 0] = 1.0
        else:
            transition_matrix[s, 0, s - 1, 1] = 1.0

        if s == columns - 2:
            transition_matrix[s, 1, s + 1, 2] = 1.0
        else:
            transition_matrix[s, 1, s + 1, 1] = 1.0
    terminal_states = [states[0],states[-1]]
    lw_env_data = {
        "states":states,
        "actions":actions,
        "rewards":rewards,
        "terminal_states":terminal_states,
        "transition_matrix":transition_matrix
    }
    lw_env = MyMDPEnv(lw_env_data)
    # lw_env = Env1()
    # columns = len(lw_env.states())
    options = list(range(start_x+block_size+int(block_size/2),start_x+int(lw_env.states()[-1])*block_size,block_size))
    x_agent=random.choice(options)
    y_agent=start_y+block_size/2
    lw_agent = LineWorldAgent(x_agent,y_agent)
    pi,Vs = policy_iteration_on_line_world(lw_env)
    while run:
        WIN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        s = int((lw_agent.x-start_x-block_size/2)/block_size)
        if lw_env.is_state_terminal(s):
            pass
        else:
            best_action = max(pi[s].items(), key= operator.itemgetter(1))[0]
            if best_action == 1:
                lw_agent.move_right(block_size)
            else:
                lw_agent.move_left(block_size)
        draw_window(lw_env,lw_agent.x,lw_agent.y)
        clock.tick(2)
            
    pygame.quit()

if __name__ == "__main__":
    main()