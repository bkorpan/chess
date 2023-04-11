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

def mcts_batched(model, roots, boards, num_simulations, batch_size, device, cpuct=1):
    for _ in range(num_simulations):
        nodes = []
        for idx in range(batch_size):
            nodes.append(select(roots[idx], boards[idx], cpuct))
        values = expand_and_evaluate_batched(nodes, boards, batch_size, model, device)
        for idx in range(batch_size):
            backpropagate(nodes[idx], values[idx], boards[idx])
    moves = []
    for idx in range(batch_size):
        max_visits = max(roots[idx].children.items(), key=lambda item: item[1].visits)[1].visits
        children_with_max_visits = list(filter(lambda item: item[1].visits == max_visits, roots[idx].children.items()))
        moves.append(random.choice(children_with_max_visits)[0])
    return moves

def mcts_async(model, roots, boards, num_simulations, num_instances, device, cpuct=1):
    nodes = roots
    policies = torch.empty((num_instances, 4096+88+2), dtype=torch.float)
    values = torch.empty((num_instances, 3), dtype=torch.float)
    for sim in range(num_simulations+1):
        for idx in range(num_instances):
            if sim > 0:
                policy = policies[idx].detach().numpy()
                value = values[idx].detach().numpy()
                legal_moves = list(boards[idx].legal_moves)
                scalar_value = value[2] - value[0]
                if not legal_moves:
                    scalar_value = -1 if board.is_checkmate() else 0
                for move in legal_moves:
                    nodes[idx].children[move] = Node(parent=nodes[idx], prior=policy[move_to_index(boards[idx], move)])
                backpropagate(nodes[idx], scalar_value, boards[idx])
            if sim < num_simulations:
                nodes[idx] = select(roots[idx], boards[idx], cpuct)
                board_tensor = torch.tensor(tokenize_board(boards[idx])).unsqueeze(0).to(device)
                policy, value = model(board_tensor)
                policy = torch.nn.Softmax(dim=-1)(policy)
                value = torch.nn.Softmax(dim=-1)(value)
                policies[idx] = policy.cpu()
                values[idx] = value.cpu()
    moves = [None]*num_instances
    for idx in range(num_instances):
        max_visits = max(roots[idx].children.items(), key=lambda item: item[1].visits)[1].visits
        children_with_max_visits = list(filter(lambda item: item[1].visits == max_visits, roots[idx].children.items()))
        moves[idx] = random.choice(children_with_max_visits)[0]
    return moves

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

def expand_and_evaluate_batched(nodes, boards, batch_size, model, device):
    tokenized_boards = []
    for idx in range(batch_size):
        tokenized_boards.append(tokenize_board(boards[idx]))
    boards_tensor = torch.tensor(tokenized_boards).to(device)
    policy, value = model(boards_tensor)
    policy = torch.nn.Softmax(dim=-1)(policy)
    value = torch.nn.Softmax(dim=-1)(value)
    policy = policy.cpu().detach().numpy()
    value = value.cpu().detach().numpy()

    values = []
    for idx in range(batch_size):
        legal_moves = list(boards[idx].legal_moves)
        if not legal_moves:
            values.append(-1 if boards[idx].is_checkmate() else 0)
        else:
            values.append(value[idx, 2] - value[idx, 0])
            for move in legal_moves:
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

def uct(node, cpuct):
    u = node.prior * np.sqrt(node.parent.visits) / (1 + node.visits)
    q = node.value()
    return q + cpuct * u

# Usage example:
# model = ...  # Your PyTorch model that returns policy and value estimations for chess
# board = chess.Board()  # Create a new chess board
# best_move = mcts(model, board, num_simulations=1000)
# board.push(best_move)  # Make the best move

