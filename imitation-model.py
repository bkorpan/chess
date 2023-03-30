import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import chess.pgn
import random
import numpy as np
from collections import defaultdict

# Pre-processing functions

def load_filtered_games(pgn_filepath):
    with open(pgn_filepath) as pgn_file:
        games = []
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


def tokenize_board(board):
    tokenized_board = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:
            piece_str = piece.symbol()
            if piece.piece_type == chess.KING and (
                    board.has_kingside_castling_rights(piece.color) or board.has_queenside_castling_rights(piece.color)):
                if piece.color == chess.WHITE:
                    piece_str = piece_str.upper() + "'"
                else:
                    piece_str = piece_str.lower() + "'"
            elif piece.piece_type == chess.ROOK:
                if (piece.color == chess.WHITE and square == chess.H1 and board.has_kingside_castling_rights(piece.color)) or \
                   (piece.color == chess.WHITE and square == chess.A1 and board.has_queenside_castling_rights(piece.color)) or \
                   (piece.color == chess.BLACK and square == chess.H8 and board.has_kingside_castling_rights(piece.color)) or \
                   (piece.color == chess.BLACK and square == chess.A8 and board.has_queenside_castling_rights(piece.color)):
                    if piece.color == chess.WHITE:
                        piece_str = piece_str.upper() + "'"
                    else:
                        piece_str = piece_str.lower() + "'"
                else:
                    if piece.color == chess.WHITE:
                        piece_str = piece_str.upper()
                    else:
                        piece_str = piece_str.lower()
            elif board.ep_square:
                if square == board.ep_square:
                    if piece.color == chess.WHITE:
                        piece_str = piece_str.upper() + "'"
                    else:
                        piece_str = piece_str.lower() + "'"
            else:
                if piece.color == chess.WHITE:
                    piece_str = piece_str.upper()
                else:
                    piece_str = piece_str.lower()
        else:
            piece_str = '0'

        tokenized_board.append(piece_str)

    return tokenized_board


def game_to_tokenized_pairs(game):
    tokenized_pairs = []
    board = game.board()

    for move in game.mainline_moves():
        tokenized_board = tokenize_board(board)
        tokenized_move = move.uci()
        tokenized_pairs.append((tokenized_board, tokenized_move))

        board.push(move)

    return tokenized_pairs


class ChessDataset(Dataset):
    def __init__(self, pgn_file, transform=None):
        self.transform = transform
        self.pgn = open(pgn_file)
        self.games = []
        while True:
            game = chess.pgn.read_game(self.pgn)
            if game is None:
                break
            self.games.append(game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        board = game.board()
        moves = list(game.mainline_moves())
        return {'board': board.fen(), 'moves': moves}


def prepare_pretraining_set(games):
    pretraining_set = []

    for game in games:
        pretraining_set.extend(game_to_tokenized_pairs(game))

    return pretraining_set


def prepare_metatraining_set(games):
    metatraining_set = defaultdict(list)

    for game in games:
        white_player = game.headers['White']
        black_player = game.headers['Black']
        game_data = game_to_tokenized_pairs(game)
        white_moves = game_data[0::2]
        black_moves = game_data[1::2]
        metatraining_set[white_player].extend(white_moves)
        metatraining_set[black_player].extend(black_moves)

    return metatraining_set


# Model classes

# Define the encoding dictionary
TOKENS = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    'R\'': 13, 'K\'': 14, 'r\'': 15, 'k\'': 16,
    'P\'': 17, 'p\'': 18, '0': 0
}

class GPTDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class ChessTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_tokens=len(TOKENS), num_positions=64, move_output_size=4100):
        super(ChessTransformer, self).__init__()
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.positional_encoding = self.create_positional_encoding(num_positions, d_model)

        decoder_layer = GPTDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.final_attention = nn.MultiheadAttention(d_model, nhead)
        self.output_layer = nn.Linear(d_model, move_output_size)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(64)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = self.token_embedding(x) * torch.sqrt(torch.tensor(self.token_embedding.embedding_dim, dtype=torch.float))
        x = x + self.positional_encoding[:x.size(0), :]
        x = self.transformer_decoder(x)

        # Pass the output of the transformer through the output layer
        x = self.output_layer(x[-1])  # Use the last token's output
        return x

# Meta training function
def maml_train(model, metatraining_set, inner_lr, outer_lr, inner_steps, num_episodes, num_support, num_query, device):
    model.train()
    outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    for episode in range(num_episodes):
        player = random.choice(list(metatraining_set.keys()))
        data = random.sample(metatraining_set[player], num_support + num_query)

        support_set = data[:num_support]
        query_set = data[num_support:]

        model_copy = copy.deepcopy(model)
        model_copy.train()
        inner_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr)

        for _ in range(inner_steps):
            for board_tokens, move_vector in support_set:
                board_tokens = torch.tensor(board_tokens, dtype=torch.long, device=device).unsqueeze(0)
                move_vector = torch.tensor(move_vector, dtype=torch.float, device=device).unsqueeze(0)

                inner_optimizer.zero_grad()
                output = model_copy(board_tokens)
                loss = nn.BCEWithLogitsLoss()(output, move_vector)
                loss.backward()
                inner_optimizer.step()

        outer_optimizer.zero_grad()
        outer_loss = 0
        for board_tokens, move_vector in query_set:
            board_tokens = torch.tensor(board_tokens, dtype=torch.long, device=device).unsqueeze(0)
            move_vector = torch.tensor(move_vector, dtype=torch.float, device=device).unsqueeze(0)

            output = model_copy(board_tokens)
            loss = nn.BCEWithLogitsLoss()(output, move_vector)
            outer_loss += loss.item()

        outer_loss /= num_query
        outer_loss.backward()
        outer_optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}: Loss = {outer_loss.item()}")


# Hyperparameters
d_model = 256
nhead = 8
num_layers = 8
dim_feedforward = 4*d_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = ChessTransformer(d_model, nhead, num_layers, dim_feedforward).to(device)

# Load and process the dataset
pgn_file = 'filtered_games.pgn'
dataset = load_filtered_games(pgn_file)

# Prepare the pretraining and metatraining sets
pretraining_set = prepare_pretraining_set(dataset)
metatraining_set = prepare_metatraining_set(dataset)

# Pretrain the model
pretrain_epochs = 100
pretrain_batch_size = 32
pretrain_lr = 1e-4

pretrain_loader = DataLoader(pretraining_set, batch_size=pretrain_batch_size, shuffle=True)
pretrain_optimizer = optim.Adam(model.parameters(), lr=pretrain_lr)

for epoch in range(pretrain_epochs):
    total_loss = 0
    for i, (board_tokens, move_vector) in enumerate(pretrain_loader):
        board_tokens = board_tokens.to(device)
        move_vector = move_vector.to(device)

        pretrain_optimizer.zero_grad()
        output = model(board_tokens)
        loss = nn.BCEWithLogitsLoss()(output, move_vector)
        loss.backward()
        pretrain_optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{pretrain_epochs}: Loss = {total_loss / len(pretrain_loader)}")

# MAML train the model
inner_lr = 1e-2
outer_lr = 1e-4
inner_steps = 10
num_episodes = 10000
num_support = 100
num_query = 100

maml_train(model, metatraining_set, inner_lr, outer_lr, inner_steps, num_episodes, num_support, num_query, device)

# Save the model
torch.save(model.state_dict(), 'chess_transformer_' + str(d_model) + '_' + str(nhead) + '_' + str(num_layers) + '.pth')

