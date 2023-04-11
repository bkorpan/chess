import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import chess
from self_play import self_play, self_play_batched, self_play_async
from models import ChessTransformer
from chess_util import SelfPlayDataset
import asyncio

async def train_self_play(model, num_rounds, num_games, num_simulations_max, epochs, lr, batch_size, device):
    num_simulations = 1
    for curr_round in range(num_rounds):
        print(f"Starting round {curr_round+1}")
        data = await self_play_async(model, num_games, num_simulations, batch_size, device)
        #data = self_play(model, num_games, num_simulations, batch_size, device)
        print(f"Num samples: {len(data)}")
        loader = DataLoader(SelfPlayDataset(data), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0
            batch_loss = 0
            for i, (board, move, outcome) in enumerate(loader):
                if i % 100 == 0 and i != 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {i}/{len(loader)}: Loss = {batch_loss / 100}")
                    batch_loss = 0
                    #scheduler.step()

                board = board.to(device)
                move = move.to(device)
                outcome = outcome.to(device)

                optimizer.zero_grad()
                policy, value = model(board)
                loss = nn.CrossEntropyLoss()(policy, move) + nn.CrossEntropyLoss()(value, outcome + 1)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_loss += loss.item()

                #if epoch == epochs-1:
                    #for j in range(batch_size):
                    #    np = 0
                    #    print(f"Sample {j}:\n")
                    #    for k in range(20480):
                    #        if output[j, k] > 0:
                    #            print(f"P({k} | {j}) = {output[j, k]}\n")
                    #print(attn.shape)
                    #print(attn[0])

            print(f"Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss / len(loader)}")

        num_simulations = min(num_simulations + 3, num_simulations_max)

# Hyperparameters
d_model = 128
nhead = 8
num_layers = 8
dim_feedforward = 4*d_model

# Self-play parameters
num_rounds = 100
num_games = 512
num_simulations_max = 100

# Training parameters
epochs = 1
lr = 5e-4
batch_size = 256

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda available: " + str(torch.cuda.is_available()))

torch.set_printoptions(threshold=65536)

# Initialize the model
model = ChessTransformer(device, d_model, nhead, num_layers, dim_feedforward).to(device=device, non_blocking=True)

asyncio.run(train_self_play(model, num_rounds, num_games, num_simulations_max, epochs, lr, batch_size, device))
#train_self_play(model, num_rounds, num_games, num_simulations_max, epochs, lr, batch_size, device)

torch.save(model.state_dict(), 'self_play_chess_transformer_' + str(d_model) + '_' + str(nhead) + '_' + str(num_layers) + '.pth')
