import math
import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import chess

from self_play import self_play, self_play_batched, self_play_threaded
from models import ChessTransformer
from chess_util import SelfPlayDataset, tokenize_board

def train_self_play(model, num_rounds, num_games, num_simulations, self_play_batch_size, epochs, lr, batch_size):
    scaler = GradScaler()
    for curr_round in range(num_rounds):
        print(f"Starting round {curr_round+1}")
        model.eval()
        with torch.no_grad(), autocast():
            data = self_play_batched(model, num_games, num_simulations, self_play_batch_size)
        print(f"Num samples: {len(data)}")
        model.train()
        loader = DataLoader(SelfPlayDataset(data), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0
            batch_loss = 0
            for i, (board, move, outcome) in enumerate(loader):
                if i % 100 == 0 and i != 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {i}/{len(loader)}: Loss = {batch_loss / 100}")
                    batch_loss = 0

                board = board.to(device)
                move = move.to(device)
                outcome = outcome.to(device)

                optimizer.zero_grad()
                with autocast():
                    policy, value = model(board)
                    loss = nn.CrossEntropyLoss()(policy, move) + nn.CrossEntropyLoss()(value, outcome + 1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss / len(loader)}")

parser = argparse.ArgumentParser()
parser.add_argument('--job-dir', type=str, default="", help='Path to the directory where the model checkpoints will be saved')
parser.add_argument('--checkpoint', type=int, default=0, help='Checkpoint to load')

parser.add_argument('--dmodel', type=int, default=128, help='Hidden dimensionality of decoders')
parser.add_argument('--dff', type=int, default=512, help='Dimensionality of feedforward networks')
parser.add_argument('--nheads', type=int, default=8, help='Number of split attention heads per decoder layer')
parser.add_argument('--nlayers', type=int, default=8, help='Number of decoder layers')

parser.add_argument('--nrounds', type=int, default=10, help='Number of rounds of self play')
parser.add_argument('--ngames', type=int, default=512, help='Number of games per round of self play')
parser.add_argument('--nsims', type=int, default=1600, help='Number of simulations per move during self play')
parser.add_argument('--self-play-batch-size', type=int, default=256, help='Batch size used for model calls during self play')

parser.add_argument('--epochs', type=int, default=1, help='Epochs of training per round of self play')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=256, help='Batch size used during training')

args = parser.parse_args()

model_dir = args.job_dir
checkpoint_number = args.checkpoint

# Hyperparameters
d_model = args.dmodel
nhead = args.nheads
num_layers = args.nlayers
dim_feedforward = args.dff

# Self-play parameters
num_rounds = args.nrounds
num_games = args.ngames
num_simulations = args.nsims
self_play_batch_size = args.self_play_batch_size

# Training parameters
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda available: " + str(torch.cuda.is_available()))

torch.set_printoptions(threshold=65536)

sys.setrecursionlimit(10000)

# Initialize the model
model = ChessTransformer(device, d_model, nhead, num_layers, dim_feedforward).to(device)

# Load checkpoint, if applicable
if checkpoint_number > 0:
    checkpoint_path = os.path.join(model_dir, 'chess_transformer_' + str(d_model) + '_' + str(dim_feedforward) + '_' + str(nhead) + '_' + str(num_layers) + '_' + str(checkpoint_number) + '.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

train_self_play(model, num_rounds, num_games, num_simulations, self_play_batch_size, epochs, lr, batch_size)

checkpoint_path = os.path.join(model_dir, 'chess_transformer_' + str(d_model) + '_' + str(dim_feedforward) + '_' + str(nhead) + '_' + str(num_layers) + '_' + str(checkpoint_number+1) + '.pth')
torch.save(model.state_dict(), checkpoint_path)
