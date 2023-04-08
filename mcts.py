import torch
import chess
import random
import numpy as np
from collections import defaultdict
from chess_util import tokenize_board, move_to_index

class Node:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits


def mcts(model, root, board, num_simulations, device, cpuct=1):
    for _ in range(num_simulations):
        node = select(root, board, cpuct)
        value = expand_and_evaluate(node, board, model, device)
        backpropagate(node, value, board)
    max_visits = max(root.children.items(), key=lambda item: item[1].visits)[1].visits
    children_with_max_visits = list(filter(lambda item: item[1].visits == max_visits, root.children.items()))
    return random.choice(children_with_max_visits)[0]

def select(node, board, cpuct):
    while node.expanded():
        move, node = max(node.children.items(), key=lambda item: uct(item[1], cpuct))
        board.push(move)
    return node

def expand_and_evaluate(node, board, model, device):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return -1 if board.is_checkmate() else 0

    board_tensor = torch.tensor(tokenize_board(board)).unsqueeze(0).to(device)
    policy, value = model(board_tensor)
    policy = torch.nn.Softmax(dim=-1)(policy)
    value = torch.nn.Softmax(dim=-1)(value)
    policy = policy[0].cpu().detach().numpy()
    value = value[0].cpu().detach().numpy()

    for move in legal_moves:
        node.children[move] = Node(parent=node, prior=policy[move_to_index(board, move)])

    return value[2] - value[0]

def backpropagate(node, value, board):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent
        value = -value
        if node is not None:
            board.pop()

def uct(node, cpuct):
    u = node.prior * np.sqrt(node.parent.visits) / (1 + node.visits)
    q = node.value()
    return q + cpuct * u

# Usage example:
# model = ...  # Your PyTorch model that returns policy and value estimations for chess
# board = chess.Board()  # Create a new chess board
# best_move = mcts(model, board, num_simulations=1000)
# board.push(best_move)  # Make the best move

