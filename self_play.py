import copy
from mcts import mcts, Node
import numpy as np
import chess

def self_play(model, num_games, num_simulations, device, cpuct=1):
    game_data = []

    for game in range(num_games):
        print(f"Starting game {game+1}")

        root = Node()
        node = root
        board = chess.Board()
        game_states = []
        game_moves = []

        while not board.is_game_over():
            game_states.append(copy.deepcopy(board))
            move = mcts(model, node, board, num_simulations, device, cpuct)
            board.push(move)
            node = node.children[move]
            node.parent = None
            game_moves.append(move)

        winner_value = compute_winner_value(board)
        move_probabilities = compute_move_probabilities(root, game_moves)

        for state, probs in zip(game_states, move_probabilities):
            game_data.append((state, probs, winner_value))
            winner_value = -winner_value

    return game_data

def compute_winner_value(board):
    if board.is_checkmate():
        return -1
    return 0

def compute_move_probabilities(root, game_moves):
    move_probabilities = []
    node = root

    for game_move in game_moves:
        legal_moves = list(node.children.keys())

        move_probs = np.zeros(len(legal_moves))
        for idx, move in enumerate(legal_moves):
            move_probs[idx] = node.children[move].visits

        if (move_probs.sum() > 0):
            move_probs /= move_probs.sum()
        else:
            move_probs = np.ones(len(legal_moves)) / len(legal_moves)
        move_probabilities.append(zip(legal_moves, move_probs))

        node = node.children[game_move]

    return move_probabilities

# Usage example:
# model = ...  # Your PyTorch model that returns policy and value estimations for chess
# game_data = self_play(model, num_games=10, num_simulations=1000)

