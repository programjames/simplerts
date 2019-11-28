from game import Game
from example import RandomPlayer

import random
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pygame
pygame.init()

class NNPlayer():
    def __init__(self, size=None, discount=None, memory=None, do_random=None, rand_decrease=None, load=None):
        """
        size - size of the game board
        discount - the amount to worry about from a future reward
        memory - the max number of training points
        do_random - probability of just doing a random move (so it learns).
                    it starts at 1, and decreases each update by rand_decrease
        """
        if load is not None:
            with open(load+".json", "r") as f:
                model_json = f.read()
            self.q_model = tf.keras.models.model_from_json(model_json)
            self.q_model.load_weights(load+".h5")
            self.q_model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
            self.size, self.discount, self.x_data, self.y_data, self.space_left, self.memory, self.index, self.do_random, self.rand_decrease = pickle.load(open(load+".pkl", "rb"))
            if size is not None:
                self.size = size
            if discount is not None:
                self.discount = discount
            if memory is not None:
                self.memory = memory
            if do_random is not None:
                self.do_random = do_random
            if rand_decrease is not None:
                self.rand_decrease = rand_decrease
        else:
            size = size if size is not None else 10
            discount = discount if discount is not None else 0.8
            memory = memory if memory is not None else 1000
            do_random = do_random if do_random is not None else 1
            rand_decrease = rand_decrease if rand_decrease is not None else 0.00001
            self.q_model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(size, size)),
              tf.keras.layers.Activation('tanh'),
              tf.keras.layers.Dense(16, activation='tanh'),
              tf.keras.layers.Dense(4, activation='tanh'),
              tf.keras.layers.Dense(1, activation='tanh')
            ])
            self.q_model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
            
            self.size = size
            self.discount = discount
            self.x_data = []
            self.y_data = []
            self.space_left = memory
            self.memory = memory
            self.index = 0
            self.do_random = do_random
            self.rand_decrease = rand_decrease
    def save(self, path="model"):
        model_json = self.q_model.to_json()
        with open(path+".json", "w") as f:
            f.write(model_json)
        self.q_model.save_weights(path+".h5")
        other_info = [self.size,
                      self.discount,
                      self.x_data,
                      self.y_data,
                      self.space_left,
                      self.memory,
                      self.index,
                      self.do_random,
                      self.rand_decrease]
        pickle.dump(other_info, open(path+".pkl", "wb"))
    def new_match(self, opponent):
        self.game = Game(self.size)
        self.opponent = opponent
            
    def move(self, board):
        my_spots = []
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] > 0:
                    my_spots.append((x, y))
        if len(my_spots) == 0:
            return None
            
        if random.random() < self.do_random:
            spot = random.choice(my_spots)
            d = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            return (spot, d)
        
        best_move = None
        best_q_value = self.q_model.predict([[board]])[0][0]
        for spot in my_spots:
            for d in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                move = (spot, d)
                new_board = Game.move_in_place(board, move, self.size)
                q_value = self.q_model.predict([[new_board]])[0][0]
                if q_value > best_q_value:
                    best_move = move
                    best_q_value = q_value
        return best_move
        
    def generate_data(self):
        self.do_random -= self.rand_decrease
        old_board = self.game.copy_board()
        reward = self.game.update(self, self.opponent)
        new_q_value = self.q_model.predict([[self.game.board]])[0][0]
        target_q_value = reward + self.discount * new_q_value
        if self.space_left > 0:
            self.x_data.append(old_board)
            self.y_data.append(target_q_value)
            self.space_left -= 1
        else:
            self.x_data[self.index] = old_board
            self.y_data[self.index] = target_q_value
            self.index += 1
            self.index %= self.memory
            
        if reward != 0:
            return reward
        return False
        
class Trainer(object):
    def __init__(self, player, opponents, size=10, do_random=1, rand_decrease=0.000001):
        self.opponents = opponents
        if player is None:
            player = NNPlayer(size=size, do_random=do_random, rand_decrease=rand_decrease)
        self.player = player
        self.player.new_match(random.choice(self.opponents))
        self.wins = 0
        self.losses = 0
        
    def train(self, batch_size=None):
        r = self.player.generate_data()
        if r != 0:
            if r == 1:
                self.wins += 1
            else:
                self.losses += 1
            self.player.new_match(random.choice(self.opponents))
        self.player.q_model.fit([self.player.x_data], self.player.y_data, batch_size=batch_size, verbose=0)
    
    def draw(self):
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode((500, 500))
        w, h = self.screen.get_size()
        board = self.player.game.board
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
                self.screen.fill(c, (x*dw, y*dh, dw, dh))
        pygame.display.update()
        pygame.event.pump()
    
    def add_opponent(self, opponent):
        self.opponents.append(opponent)

if __name__ == "__main__":
    trainer = Trainer(NNPlayer(size=3), [NNPlayer(load="model", size=3)])
    epoch = 0
    win_rates = [0]
    plt.ion()
    while True:
        epoch += 1
        for i in range(1000):
            trainer.train(batch_size=10)
            wr = trainer.wins/max(1, trainer.wins+trainer.losses)
            if wr != win_rates[-1]:
                win_rates.append(wr)
                plt.plot(win_rates)
                plt.draw()
                plt.pause(0.0001)
                plt.clf()
##            trainer.draw()
        print("Chance of random move:", trainer.player.do_random)
        print("Epoch", epoch, "finished.")
        trainer.player.save("noob")
