import operator
import random
import pygame
import numpy as np
from drl_lib.to_do.MDP_contracts import MyMDPEnv
from drl_lib.to_do.dynamic_programming import (
    policy_iteration_on_grid_world,
    value_iteration_on_grid_world,
)
from drl_lib.to_do.utils import init_grid_transition
from drl_lib.to_do.gw_agent import GridWorldAgent

pygame.init()
WIDTH, HEIGHT = 700, 700
WIN = pygame.display.set_mode((HEIGHT, WIDTH))
pygame.display.set_caption("Grid world")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (124, 252, 0)
RED = (255, 0, 0)

block_size = 50
start_x = 50
start_y = 50

# y=state/grid_size = row
# x = state%grid_size = col

clock = pygame.time.Clock()


def grid_world(ncols, nrows, Vs, env, x_agent=None, y_agent=None):
    for y in range(start_y, start_y + nrows * block_size, block_size):
        for x in range(start_x, start_x + ncols * block_size, block_size):
            s = find_state(
                (x - start_x) // block_size,
                (y - start_y) // block_size,
                env.states(),
                ncols,
            )
            if s is not None:
                WIN.blit(
                    pygame.font.SysFont("Arial", 15).render(
                        str(round(Vs[s], 3)), True, BLACK
                    ),
                    (x + int(block_size / 3), y + int(block_size / 3)),
                )
            rect = pygame.Rect(x, y, block_size, block_size)
            if x == start_x + (ncols - 1) * block_size and y == start_y:
                pygame.draw.rect(WIN, RED, rect, 0)
            if (
                y == start_y + (nrows - 1) * block_size
                and x == start_x + (ncols - 1) * block_size
            ):
                pygame.draw.rect(WIN, GREEN, rect, 0)
            else:
                pygame.draw.rect(WIN, BLACK, rect, 1)
    if x_agent is not None and y_agent is not None:
        pygame.draw.circle(WIN, BLACK, (x_agent, y_agent), 10, 0)


def find_state(x, y, states, grid_size):
    for state in states:
        if x == state % grid_size and y == state // grid_size:
            return state
    return None


def draw_window(ncols, nrows, Vs, env, x_agent, y_agent):
    grid_world(ncols, nrows, Vs, env, x_agent, y_agent)
    pygame.display.flip()


def main():
    run = True
    grid_size = 5
    nb_cells = grid_size * grid_size
    states = np.arange(nb_cells)
    actions = np.array([0, 1, 2, 3])  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
    rewards = np.array([-1.0, 0.0, 1.0])
    transition_matrix = init_grid_transition(grid_size, states, actions, rewards)

    terminal_states = [states[grid_size - 1], states[nb_cells - 1]]

    env_data = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "terminal_states": terminal_states,
        "transition_matrix": transition_matrix,
    }
    gw_env = MyMDPEnv(env_data)
    pi, Vs = value_iteration_on_grid_world(gw_env)
    print(Vs)

    options = [
        (x, y)
        for x in range(
            start_x + int(block_size / 2),
            start_x + grid_size * block_size + int(block_size / 2),
            block_size,
        )
        for y in range(
            start_y + int(block_size / 2),
            start_y + grid_size * block_size + int(block_size / 2),
            block_size,
        )
    ]
    options.remove(
        (
            start_x + (grid_size - 1) * block_size + int(block_size / 2),
            start_y + int(block_size / 2),
        )
    )
    options.remove(
        (
            start_x + (grid_size - 1) * block_size + int(block_size / 2),
            start_y + (grid_size - 1) * block_size + int(block_size / 2),
        )
    )
    idx = random.choice(range(len(options)))
    (x_agent, y_agent) = options[idx]
    gw_agent = GridWorldAgent(x_agent, y_agent)

    while run:
        WIN.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        s = find_state(
            (gw_agent.x - start_x - int(block_size / 2)) // block_size,
            (gw_agent.y - start_y - int(block_size / 2)) // block_size,
            gw_env.states(),
            grid_size,
        )
        if s is not None:
            if gw_env.is_state_terminal(s):
                pass
            else:
                best_action = max(pi[s].items(), key=operator.itemgetter(1))[0]
                if best_action == 0:
                    gw_agent.move_up(block_size)
                elif best_action == 1:
                    gw_agent.move_down(block_size)
                elif best_action == 2:
                    gw_agent.move_left(block_size)
                else:
                    gw_agent.move_right(block_size)

        draw_window(grid_size, grid_size, Vs, gw_env, gw_agent.x, gw_agent.y)
        clock.tick(1)

    pygame.quit()


if __name__ == "__main__":
    main()
