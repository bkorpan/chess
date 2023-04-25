import copy
from mcts import mcts, mcts_batched, Node
import numpy as np
import chess
from concurrent import futures

def self_play(model, num_games, num_simulations, cpuct=1):
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
            move = mcts(model, node, board, num_simulations, cpuct)
            board.push(move)
            node = node.children[move]
            node.parent = None
            game_moves.append(move)

        winner_value = compute_winner_value(board)
        move_probabilities = compute_move_probabilities(root, game_moves)

        for state, probs in zip(game_states, move_probabilities):
            game_data.append((state, probs, winner_value))
            winner_value = -winner_value

        root.deconstruct()

    return game_data

def self_play_threaded(model, num_games, num_simulations, num_threads, batch_size, cpuct=1):
    game_data = []
    tasks = []

    with futures.ThreadPoolExecutor() as executor:
        for _ in range(num_threads):
            tasks.append(executor.submit(self_play_batched, model, num_games, num_simulations, batch_size))

    for thread_idx in range(num_threads):
        game_data.extend(tasks[thread_idx].result())

    return game_data

def self_play_batched(model, num_games, num_simulations, batch_size, cpuct=1):
    game_data = []
    completed_games = 0
    in_progress_games = min(num_games, batch_size)

    roots = [Node() for _ in range(batch_size)]
    nodes = roots.copy()
    boards = [chess.Board() for _ in range(batch_size)]
    game_states = [[] for _ in range(batch_size)]
    game_moves = [[] for _ in range(batch_size)]

    while completed_games < num_games:
        #print(f"In progress games = {in_progress_games}")
        moves = mcts_batched(model, nodes, boards, num_simulations, in_progress_games, cpuct, randomize=True)

        for idx in range(in_progress_games):
            game_states[idx].append(copy.deepcopy(boards[idx]))
            boards[idx].push(moves[idx])
            nodes[idx] = nodes[idx].children[moves[idx]]
            nodes[idx].parent = None
            game_moves[idx].append(moves[idx])

            if boards[idx].is_game_over():
                winner_value = compute_winner_value(boards[idx])
                move_probabilities = compute_move_probabilities(roots[idx], game_moves[idx])
                for state, probs in zip(game_states[idx], move_probabilities):
                    game_data.append((state, probs, winner_value))
                    winner_value = -winner_value

                roots[idx].deconstruct()
                roots[idx] = Node()
                nodes[idx] = roots[idx]
                boards[idx] = chess.Board()
                game_states[idx] = []
                game_moves[idx] = []

                completed_games += 1
                if completed_games % 100 == 0:
                    print(f"Completed {completed_games} games")

        updated_in_progress_games = min(num_games - completed_games, batch_size)
        if updated_in_progress_games < in_progress_games:
            roots_tmp = []
            nodes_tmp = []
            boards_tmp = []
            game_states_tmp = []
            game_moves_tmp = []

            for idx in range(in_progress_games):
                if len(game_states[idx]) > 0:
                    roots_tmp.append(roots[idx])
                    nodes_tmp.append(nodes[idx])
                    boards_tmp.append(boards[idx])
                    game_states_tmp.append(game_states[idx])
                    game_moves_tmp.append(game_moves[idx])

            while len(roots_tmp) < updated_in_progress_games:
                roots_tmp.append(Node())
                nodes_tmp.append(roots_tmp[-1])
                boards_tmp.append(chess.Board())
                game_states_tmp.append([])
                game_moves_tmp.append([])

            roots = roots_tmp
            nodes = nodes_tmp
            boards = boards_tmp
            game_states = game_states_tmp
            game_moves = game_moves_tmp

            in_progress_games = updated_in_progress_games

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
