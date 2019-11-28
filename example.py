from game import Game

import random

class RandomPlayer(object):
    def __init__(self):
        pass
    def move(self, board):
        my_spots = []
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] > 0:
                    my_spots.append((x, y))
        if len(my_spots) == 0:
            return None
        spot = random.choice(my_spots)
        d = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        return (spot, d)
        
        
if __name__ == "__main__":
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
    
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    game = Game()
    screen = pygame.display.set_mode((500, 500))
    winner = 0
    while winner == 0:
        draw(game.board, screen)
        pygame.display.update()
        winner = game.update(player1, player2)
    print(game.turn)