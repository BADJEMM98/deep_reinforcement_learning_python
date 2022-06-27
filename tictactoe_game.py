from random import choice, choices
from tracemalloc import start
import pygame
import numpy as np
from drl_lib.to_do.tictactoe_env import TicTacToeEnv
from drl_lib.to_do.monte_carlo_methods import monte_carlo_es_on_tic_tac_toe_solo, off_policy_monte_carlo_control_on_tic_tac_toe_solo, on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo

pygame.init()
WIDTH, HEIGHT = 470, 470
LINEWIDTH = 10
CIRCLE_RADIUS = 55
CIRCLE_WIDTH = 15

CROSS_WIDTH =15
WIN = pygame.display.set_mode((HEIGHT, WIDTH))
pygame.display.set_caption("tic Tac Toe")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (124, 252, 0)
RED = (255, 0, 0)
BGCOLOR = (255, 121, 108)
LINECOLOR  = (173,65,54)
CROSS_COLOR = (84, 84, 84)
CIRCLE_COLOR = (242, 235, 211)

BLOCKHEIGHT = 150
BLOCKWIDTH = 150


# y=state/grid_size = row
# x = state%grid_size = col

clock = pygame.time.Clock()

def draw_grid():
    #Horizontal lines
    pygame.draw.line(WIN,LINECOLOR,(0,BLOCKHEIGHT),(WIDTH,BLOCKHEIGHT),LINEWIDTH)
    pygame.draw.line(WIN,LINECOLOR,(0,2*BLOCKHEIGHT+LINEWIDTH),(WIDTH,2*BLOCKHEIGHT+LINEWIDTH),LINEWIDTH)

    #vertical lines
    pygame.draw.line(WIN,LINECOLOR,(BLOCKWIDTH,0),(BLOCKWIDTH,HEIGHT),LINEWIDTH)
    pygame.draw.line(WIN,LINECOLOR,(2*BLOCKWIDTH+LINEWIDTH,0),(2*BLOCKHEIGHT+LINEWIDTH,HEIGHT),LINEWIDTH)
 
def draw_figures(board,size):
    for row in range(size):
        for col in range(size):
            if board[row][col]==1:
                x = col*BLOCKWIDTH+BLOCKWIDTH/2
                y = row*BLOCKHEIGHT+BLOCKHEIGHT/2
                pygame.draw.circle(WIN,CIRCLE_COLOR,(x,y),CIRCLE_RADIUS,CIRCLE_WIDTH)
            if board[row][col]==2:
                eps=35
                pygame.draw.line(WIN,CROSS_COLOR,(col*BLOCKWIDTH+eps,row*BLOCKHEIGHT+BLOCKHEIGHT-eps),(col*BLOCKWIDTH+BLOCKWIDTH-eps,row*BLOCKWIDTH+eps),CROSS_WIDTH)
                pygame.draw.line(WIN,CROSS_COLOR,(col*BLOCKWIDTH+eps,row*BLOCKHEIGHT+eps),(col*BLOCKWIDTH+BLOCKWIDTH-eps,row*BLOCKWIDTH +BLOCKWIDTH-eps),CROSS_WIDTH)


def draw_window(board,size):
    WIN.fill(BGCOLOR)
    clock.tick(3)
    draw_grid()
    draw_figures(board,size)
    pygame.display.flip()

def find_action(x, y, actions, grid_size):
    for action in actions:
        if x == action % grid_size and y == action // grid_size:
            return action
    return None


def main():
    run=True
    env = TicTacToeEnv()
    pi_and_q = off_policy_monte_carlo_control_on_tic_tac_toe_solo()
    # off_policy_monte_carlo_control_on_tic_tac_toe_solo()
    #on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo()
    #  monte_carlo_es_on_tic_tac_toe_solo()
    cpt = 0
    nb_parties = 100
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        while not env.is_game_over():
            s = env.state_id()
            # print(s)
            pis = [pi_and_q.pi[s][a] for a in env.available_actions_ids()]
            if max(pis) == 0.0:
                a = choice(env.available_actions_ids())
            else:
                a = choices(env.available_actions_ids(), weights=pis)[0]

            # faire jouer player[1]
            env.act_with_action_id(env.players[1].sign,a)
            draw_window(env.board,env.size)

            # faire jouer player[0]
            # if not env.is_game_over():
            #     action = None
            #     while True:
            #         for event in pygame.event.get():
            #             if event.type == pygame.MOUSEBUTTONDOWN:
            #                 col = event.pos[0]//BLOCKWIDTH
            #                 row = event.pos[1]//BLOCKHEIGHT
            #                 action = find_action(col,row,env.available_actions_ids(),env.size)
            #         if action is not None:
            #             break
            #     env.act_with_action_id(env.players[0].sign,action)
            #     draw_window(env.board,env.size)
            if not env.is_game_over():
                rand_action = env.players[0].play(env.available_actions_ids())
                env.act_with_action_id(env.players[0].sign,rand_action)
                draw_window(env.board,env.size)

        env.is_game_over()

        if env.players[1].is_winner:
            # print(env.players[1].is_winner)
            cpt+=1
        print("random player is winner : ",env.players[0].is_winner)
        print("Agent is winner : ",env.players[1].is_winner)
        env.reset()
        draw_window(env.board,env.size)
        nb_parties-=1
        if nb_parties == 0:
            run=False
        print("pourcentage de réussite : " ,cpt,"%")  

        

    pygame.quit()


if __name__ == "__main__":
    main()
