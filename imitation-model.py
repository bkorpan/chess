import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import chess.pgn
import random
import numpy as np
from collections import defaultdict

# Pre-processing functions

def load_games(pgn_filepath):
    with open(pgn_filepath) as pgn_file:
        games = []
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def load_n_games(pgn_filepath, n):
    with open(pgn_filepath) as pgn_file:
        games = []
        for i in range(n):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


def tokenize_board(board):
    tokenized_board = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        piece_token = 0

        if piece:
            piece_token = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
            if piece.piece_type == chess.KING and (
                    board.has_kingside_castling_rights(piece.color) or board.has_queenside_castling_rights(piece.color)):
                piece_token = 15 if piece.color == chess.WHITE else 18
            elif piece.piece_type == chess.ROOK:
                if (piece.color == chess.WHITE and square == chess.H1 and board.has_kingside_castling_rights(piece.color)) or \
                   (piece.color == chess.WHITE and square == chess.A1 and board.has_queenside_castling_rights(piece.color)) or \
                   (piece.color == chess.BLACK and square == chess.H8 and board.has_kingside_castling_rights(piece.color)) or \
                   (piece.color == chess.BLACK and square == chess.A8 and board.has_queenside_castling_rights(piece.color)):
                    piece_token = 14 if piece.color == chess.WHITE else 17
            elif board.ep_square:
                if square == board.ep_square:
                    piece_token = 13 if piece.color == chess.WHITE else 16

        tokenized_board.append(piece_token)
    return tokenized_board


def game_to_tokenized_pairs(game):
    tokenized_pairs = []
    board = game.board()

    for move in game.mainline_moves():
        tokenized_board = tokenize_board(board)
        tokenized_move = 64*move.from_square + move.to_square if not move.promotion else 4096*(move.promotion - 1) + 64*move.from_square + move.to_square
        tokenized_pairs.append((tokenized_board, tokenized_move))

        board.push(move)

    return tokenized_pairs


class ChessDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (board, move) = self.data[idx]
        return torch.tensor(board), torch.tensor(move)


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

class GPTDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class ChessTransformer(nn.Module):
    def __init__(self, device, d_model, nhead, num_layers, dim_feedforward, num_tokens=19, move_output_size=20480):
        super(ChessTransformer, self).__init__()
        assert(d_model % 128 == 0)
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.output_layer = nn.Linear(d_model, move_output_size)
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_tokens = num_tokens
        self.device = device
        self.positional_encoding = self.positional_encoding_2d(d_model)
        self.one_hot_positional_encoding = nn.Parameter(nn.functional.one_hot(torch.arange(0,64), 64).repeat(1, d_model // 128).type(torch.float).to(device), requires_grad=False)
        self.decoders = []
        for i in range(num_layers):
            self.decoders.append(GPTDecoderLayer(d_model, nhead, dim_feedforward).to(device))

    def positional_encoding_2d(self, num_features, chessboard_size=(8, 8)):
        assert num_features % 4 == 0, "num_features should be divisible by 4."

        height, width = chessboard_size
        encoding = torch.zeros(height, width, num_features)

        # Compute the row (rank) and column (file) position tensors
        row_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1).repeat(1, width).unsqueeze(2)
        col_position = torch.arange(0, width, dtype=torch.float).unsqueeze(0).repeat(height, 1).unsqueeze(2)

        # Compute the divisors for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, num_features // 2, 2).float() * (-torch.log(torch.tensor(100.)) / num_features))
        div_term = div_term.view(1, 1, num_features // 4)

        # Apply the sinusoidal functions to the row and column position tensors
        encoding[:, :, 0::4] = torch.sin(row_position * div_term)
        encoding[:, :, 1::4] = torch.cos(row_position * div_term)
        encoding[:, :, 2::4] = torch.sin(col_position * div_term)
        encoding[:, :, 3::4] = torch.cos(col_position * div_term)

        return nn.Parameter(encoding.view(height*width, num_features).to(self.device), requires_grad=False)

    def forward(self, x):
        #x = self.token_embedding(x) + self.positional_encoding
        x = nn.functional.one_hot(x, self.num_tokens).repeat(1, 1, (d_model // 2) // self.num_tokens).type(torch.float)
        tmp = nn.functional.pad(x, pad=(0, (d_model // 2) - x.shape[2]))
        x = torch.zeros(x.shape[0], x.shape[1], d_model).type(torch.float).to(device)
        for i in range (tmp.shape[0]):
            x[i] = torch.cat((tmp[i], self.one_hot_positional_encoding), dim=1)

        for i in range(num_layers):
            x = self.decoders[i](x)

        # Pass the output of the transformer through the output layer
        x = self.output_layer(x[:,-1])  # Use the last token's output
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
            for board, move in support_set:
                board = torch.tensor(board, dtype=torch.long, device=device)
                move = nn.functional.one_hot(torch.tensor(move), 20480).type(torch.float).to(device)

                inner_optimizer.zero_grad()
                output = model_copy(board)
                loss = nn.CrossEntropyLoss()(output, move)
                loss.backward()
                inner_optimizer.step()

        outer_optimizer.zero_grad()
        outer_loss = 0
        for board, move in query_set:
            board = torch.tensor(board, dtype=torch.long, device=device)
            move = nn.functional.one_hot(torch.tensor(move), 20480).type(torch.float).to(device)

            output = model_copy(board)
            loss = nn.CrossEntropyLoss()(output, move)
            outer_loss += loss.item()

        outer_loss /= num_query
        outer_loss.backward()
        outer_optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}: Loss = {outer_loss.item()}")


# Hyperparameters
d_model = 128
nhead = 4
num_layers = 8
dim_feedforward = 4*d_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda available: " + str(torch.cuda.is_available()))

torch.set_printoptions(threshold=65536)

# Initialize the model
model = ChessTransformer(device, d_model, nhead, num_layers, dim_feedforward).to(device)

# Load and process the dataset
pgn_file = 'filtered_games.pgn'
games = load_n_games(pgn_file, 1)

# Prepare the pretraining and metatraining sets
pretraining_set = prepare_pretraining_set(games)
metatraining_set = prepare_metatraining_set(games)

# Pretrain the model
pretrain_epochs = 10000
pretrain_batch_size = 13
pretrain_lr = 5e-4

pretrain_loader = DataLoader(ChessDataset(pretraining_set), batch_size=pretrain_batch_size, shuffle=True)
pretrain_optimizer = optim.SGD(model.parameters(), lr=pretrain_lr)

print("Starting pretraining")

for epoch in range(pretrain_epochs):
    total_loss = 0
    batch_loss = 0
    for i, (board, move) in enumerate(pretrain_loader):
        if i % 100 == 0 and i != 0:
            print(f"Epoch {epoch + 1}/{pretrain_epochs}, Batch {i}/{len(pretrain_loader)}: Loss = {batch_loss / 100}")
            batch_loss = 0
        
        board = board.to(device)
        move = move.to(device)

        pretrain_optimizer.zero_grad()
        output = model(board)
        loss = nn.CrossEntropyLoss()(output, move)
        loss.backward()
        pretrain_optimizer.step()

        total_loss += loss.item()
        batch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{pretrain_epochs}: Loss = {total_loss / len(pretrain_loader)}")

print("Pretraining complete!")

# MAML train the model
#inner_lr = 1e-3
#outer_lr = 1e-4
#inner_steps = 2
#num_episodes = 1000
#num_support = 500
#num_query = 500

#print("Starting metatraining")

#maml_train(model, metatraining_set, inner_lr, outer_lr, inner_steps, num_episodes, num_support, num_query, device)

# Save the model
torch.save(model.state_dict(), 'chess_transformer_' + str(d_model) + '_' + str(nhead) + '_' + str(num_layers) + '.pth')

