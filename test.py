from nnplayer import NNPlayer
from example import RandomPlayer
from game import Game

import pygame
import numpy as np
pygame.init()

def draw(board, screen):
    w, h = screen.get_size()
    s = len(board)
    dw = w//s
    dh = h//s
    for x in range(s):
        for y in range(s):
            if board[x][y] < 0:
                v = 255 - int(255/(1 + np.exp(board[x][y])))
                c = (v, v, 255)
            elif board[x][y] > 0:
                v = 255 - int(255/(1 + np.exp(-board[x][y])))
                c = (255, v, v)
            else:
                c = (255, 255, 255)
            screen.fill(c, (x*dw, y*dh, dw, dh))

while True:
    player1 = RandomPlayer()
    player2 = NNPlayer(load="model")
    game = Game(size=3)
    screen = pygame.display.set_mode((500, 500))
    winner = 0
    while winner == 0:
        draw(game.board, screen)
        pygame.display.update()
        winner = game.update(player1, player2)
