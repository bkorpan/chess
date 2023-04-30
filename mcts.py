import torch
import chess
import random
import numpy as np
from collections import defaultdict
from chess_util import tokenize_board, move_to_index
import time

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

    def deconstruct(self):
        self.parent = None
        for item in self.children.items():
            item[1].deconstruct()
        self.children.clear()

def mcts(model, root, board, num_simulations, cpuct=1, exp=1.1):
    for _ in range(num_simulations):
        node = select(root, board, cpuct, exp)
        value = expand_and_evaluate(node, board, model)
        backpropagate(node, value, board)
    max_visits = max(root.children.items(), key=lambda item: item[1].visits)[1].visits
    children_with_max_visits = list(filter(lambda item: item[1].visits == max_visits, root.children.items()))
    return random.choice(children_with_max_visits)[0]

def mcts_batched(model, roots, boards, num_simulations, batch_size, cpuct=1, exp=1.1, sample_moves=False):
    for _ in range(num_simulations+1):
        nodes = []
        for idx in range(batch_size):
            nodes.append(select(roots[idx], boards[idx], cpuct, exp))
        values = expand_and_evaluate_batched(nodes, boards, batch_size, model)
        for idx in range(batch_size):
            backpropagate(nodes[idx], values[idx], boards[idx])
    moves = []
    if sample_moves:
        for idx in range(batch_size):
            choice = random.randint(0, num_simulations-1)
            for item in roots[idx].children.items():
                visits = item[1].visits
                if choice < visits:
                    moves.append(item[0])
                    break
                choice -= visits
    else:
        for idx in range(batch_size):
            max_visits = max(roots[idx].children.items(), key=lambda item: item[1].visits)[1].visits
            children_with_max_visits = list(filter(lambda item: item[1].visits == max_visits, roots[idx].children.items()))
            moves.append(random.choice(children_with_max_visits)[0])
    return moves

def select(node, board, cpuct, exp):
    while node.expanded():
        if not board.is_game_over():
            move, node = max(node.children.items(), key=lambda item: uct(item[1], cpuct, exp))
            board.push(move)
        else:
            break
    return node

def expand_and_evaluate(node, board, model):
    legal_moves = list(board.legal_moves)
    if board.is_game_over():
        return -1 if board.is_checkmate() else 0

    board_tensor = torch.tensor(tokenize_board(board)).unsqueeze(0).to(model.device)
    policy, value = model(board_tensor)
    policy = torch.nn.Softmax(dim=-1)(policy)
    value = torch.nn.Softmax(dim=-1)(value)
    policy = policy[0].cpu().detach().numpy()
    value = value[0].cpu().detach().numpy()

    prior_sum = sum(map(lambda move: policy[move_to_index(board, move)], legal_moves))
    if prior_sum == 0:
        prior_sum = 1

    for move in legal_moves:
        node.children[move] = Node(parent=node, prior=policy[move_to_index(board, move)]/prior_sum)

    return value[2] - value[0]

def expand_and_evaluate_batched(nodes, boards, batch_size, model):
    tokenized_boards = []
    for idx in range(batch_size):
        tokenized_boards.append(tokenize_board(boards[idx]))
    boards_tensor = torch.tensor(tokenized_boards).to(model.device)
    policy, value = model(boards_tensor)
    policy = torch.nn.Softmax(dim=-1)(policy)
    value = torch.nn.Softmax(dim=-1)(value)
    policy = policy.cpu().detach().numpy()
    value = value.cpu().detach().numpy()

    values = []
    for idx in range(batch_size):
        if boards[idx].is_game_over():
            values.append(-1 if boards[idx].is_checkmate() else 0)
        else:
            values.append(value[idx, 2] - value[idx, 0])
            for move in boards[idx].legal_moves:
                nodes[idx].children[move] = Node(parent=nodes[idx], prior=policy[idx, move_to_index(boards[idx], move)])

    return values

def backpropagate(node, value, board):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent
        value = -value
        if node is not None:
            board.pop()

def uct(node, cpuct, exp):
    u = node.prior / (1 + node.visits)**exp * np.sqrt(node.parent.visits)
    q = node.value()
    return q + cpuct * u
