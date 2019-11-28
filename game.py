class Game(object):
    # player1 = positive number
    # player2 = negative number
    def __init__(self, size=10, gen_rate=0.01, max_turns=10000):
        """
        size = the board consists of size x size squares.
        gen_rate = the regeneration rate of each occupied square on the board.
        """
        self.size = size
        self.board = [[0 for i in range(size)] for j in range(size)]
        self.board[0][0] = 1
        self.board[-1][-1] = -1
        self.gen_rate = gen_rate
        self.turn = 0
        self.max_turns = 10000
    
    def update(self, player1, player2):
        self.turn += 1
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] < 0:
                    self.board[x][y] -= self.gen_rate
                elif self.board[x][y] > 0:
                    self.board[x][y] += self.gen_rate
        # Make player1's move.
        move = player1.move(self.board)
        if move is not None:
            x, y = move[0]
            dx, dy = move[1]
            if dx**2 + dy**2 < 2 and self.board[x][y] > 0:
                x2, y2 = x + dx, y + dy
                self.board[x][y] /= 2
                self.board[x2%self.size][y2%self.size] += self.board[x][y]
        # Make player2's move.
        move = player2.move([[-v for v in r] for r in self.board])
        if move is not None:
            x, y = move[0]
            dx, dy = move[1]
            if dx**2 + dy**2 < 2 and self.board[x][y] < 0:
                x2, y2 = x + dx, y + dy
                self.board[x][y] /= 2
                self.board[x2%self.size][y2%self.size] += self.board[x][y]
                
        if all(all(v >= 0 for v in r) for r in self.board):
            return 1 # player 1 wins
        elif all(all(v <= 0 for v in r) for r in self.board):
            return -1 # player 2 wins
        else:
            if self.turn > self.max_turns:
                if sum(sum(r) for r in self.board) > 0:
                    return 1
                return -1
            return 0
            
    def copy_board(self):
        # Returns a copy of the board.
        return [r[:] for r in self.board]
    
    @staticmethod
    def move_in_place(board, move, size):
        board = [r[:] for r in board] # deepcopy it
        if move is not None:
            x, y = move[0]
            dx, dy = move[1]
            if dx**2 + dy**2 < 2 and board[x][y] > 0:
                x2, y2 = x + dx, y + dy
                board[x][y] /= 2
                board[x2%size][y2%size] += board[x][y]
        return board
            
def play(player1, player2, max_turn = None):
    winner = 0
    game = Game()
    while winner == 0 and game.turn < max_turn:
        winner = game.update(player1, player2)
    return winner
