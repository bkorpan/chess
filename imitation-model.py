import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from chess import pgn
import chess
from collections import defaultdict
from typing import List, Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# Pre-processing functions

def create_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=32000)
    tokenizer.post_processor = TemplateProcessing(
        single="[UNK] $A",
        special_tokens=[("[UNK]", tokenizer.token_to_id("[UNK]"))],
    )

    tokenizer.train_from_iterator(generate_tokenizer_data(), trainer=trainer)
    return tokenizer


def generate_tokenizer_data():
    for board in chess.Board().legal_boards():
        yield encode_board(board)
        for move in board.legal_moves:
            yield move.uci()


def encode_board(board: chess.Board) -> str:
    return "\n".join(
        "".join(board.piece_at(chess.square(rank, file)) for file in range(8))
        for rank in range(7, -1, -1)
    )


def load_filtered_games(pgn_filepath):
    with open(pgn_filepath) as pgn_file:
        games = []
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def parse_game(game: chess.pgn.Game) -> List[Tuple[chess.Board, chess.Move]]:
    result = []
    node = game
    while not node.is_end():
        board = node.board()
        move = node.variations[0].move
        result.append((board, move))
        node = node.variations[0]

    return result

def parse_game(game: chess.pgn.Game, player: str) -> List[Tuple[chess.Board, chess.Move]]:
    moves = []
    board = game.board()

    for move in game.mainline_moves():
        if board.turn == chess.WHITE and player == game.headers["White"]:
            moves.append((board.copy(), move))
        elif board.turn == chess.BLACK and player == game.headers["Black"]:
            moves.append((board.copy(), move))
        board.push(move)

    return moves

# Model creation function
def create_transformer_model(input_dim, hidden_dim, output_dim, num_heads, num_layers, dff):
    inputs = Input(shape=(None,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(input_dim, hidden_dim)(inputs)

    for _ in range(num_layers):
        x = tf.keras.layers.MultiHeadAttention(num_heads, hidden_dim // num_heads)(x, x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.PositionWiseFeedForward(hidden_dim, dff)(x)
        x = tf.keras.layers.LayerNormalization()(x)

    x = Dense(output_dim, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Pre-training functions

def generate_pretraining_data(filtered_games):
    pretraining_data = []

    for game in filtered_games:
        pretraining_data.extend(parse_game(game))

    return pretraining_data


def pretrain(model, pretraining_data, epochs, batch_size):
    tokenizer = create_tokenizer()
    input_dim = len(tokenizer.get_vocab())

    optimizer = Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(epochs):
        np.random.shuffle(pretraining_data)

       	for i in range(0, len(pretraining_data), batch_size):
            batch_data = pretraining_data[i:i + batch_size]

            input_seqs = [tokenizer.tokenize(state).to_tensor() for state, move in batch_data]
            input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding="post")
            target_seqs = [tokenizer.tokenize(move.uci()).to_tensor() for state, move in batch_data]
            target_seqs = tf.keras.preprocessing.sequence.pad_sequences(target_seqs, padding="post")

            with tf.GradientTape() as tape:
                predictions = model(input_seqs, training=True)
                loss = loss_fn(target_seqs, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.numpy()}")

# Meta-training functions

def inner_loop(model, support_set, inner_lr):
    optimizer = Adam(learning_rate=inner_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    updated_weights = model.get_weights()

    for state, move in support_set:
        input_seq = tokenizer.tokenize(encode_board(state)).to_tensor()[tf.newaxis, :]
        target_seq = tokenizer.tokenize(move.uci()).to_tensor()[tf.newaxis, :]

        with tf.GradientTape() as tape:
            predictions = model(input_seq, training=True)
            loss = loss_fn(target_seq, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        updated_weights = [w - inner_lr * g for w, g in zip(updated_weights, gradients)]

    return updated_weights


def meta_train(model, games_by_player, epochs, k_shot_support, k_shot_query, meta_lr, inner_lr):
    optimizer = Adam(learning_rate=meta_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(epochs):
        player = np.random.choice(list(games_by_player.keys()))

        player_data = []
        for game in games_by_player[player]:
            player_moves = parse_game(game, player)
            player_data.extend(player_moves)

        np.random.shuffle(player_data)

        support_set = player_data[:k_shot_support]
        query_set = player_data[k_shot_support:k_shot_support + k_shot_query]

        updated_weights = inner_loop(model, support_set, inner_lr)
        model.set_weights(updated_weights)

        input_seqs = [tokenizer.tokenize(state).to_tensor() for state, move in query_set]
        input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, padding="post")
        target_seqs = [tokenizer.tokenize(move.uci()).to_tensor() for state, move in query_set]
        target_seqs = tf.keras.preprocessing.sequence.pad_sequences(target_seqs, padding="post")

        with tf.GradientTape() as tape:
            predictions = model(input_seqs, training=True)
            loss = loss_fn(target_seqs, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Meta-training Epoch {epoch + 1}/{epochs}: Loss = {loss.numpy()}")


# Training the model
script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'filtered_games.pgn'
file_path = os.path.join(script_dir, file_name)
filtered_games = load_filtered_games(file_path)

pretraining_data = generate_pretraining_data(filtered_games)

# Model parameters
input_dim = len(tokenizer.get_vocab())
hidden_dim = 128
output_dim = len(tokenizer.get_vocab())
num_heads = 4
num_layers = 2
dff = 512

model = create_transformer_model(input_dim, hidden_dim, output_dim, num_heads, num_layers, dff)

pretrain(model, pretraining_data, epochs=20, batch_size=32)

games_by_player = defaultdict(list)
for game in filtered_games:
    white_player = game.headers['White']
    black_player = game.headers['Black']
    games_by_player[white_player].append(game)
    games_by_player[black_player].append(game)

meta_train(model, filtered_players, epochs=100, k_shot_support=50, k_shot_query = 50, meta_lr=1e-3, inner_lr=1e-2)

