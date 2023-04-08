import torch
import torch.nn as nn

class GPTDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)

    def forward(self, tgt):
        tgt2, attn = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class ChessTransformer(nn.Module):
    def __init__(self, device, d_model, nhead, num_layers, dim_feedforward, num_tokens=19, move_output_size=4096+88+2):
        super(ChessTransformer, self).__init__()
        assert(d_model % 128 == 0)
        self.token_embedding = nn.Embedding(num_tokens, d_model)
        self.reduce_output = nn.Linear(d_model*64, d_model)
        self.policy_activation = nn.ReLU()
        self.value_activation = nn.ReLU()
        self.policy_layer = nn.Linear(d_model, dim_feedforward)
        self.value_layer = nn.Linear(d_model, dim_feedforward)
        self.policy_out = nn.Linear(dim_feedforward, move_output_size)
        self.value_out = nn.Linear(dim_feedforward, 3)
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_tokens = num_tokens
        self.device = device
        self.positional_encoding = self.positional_encoding_2d(d_model)
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
        x = self.token_embedding(x) + self.positional_encoding

        for i in range(self.num_layers):
            x = self.decoders[i](x)

        # Pass the output of the transformer through the output layer
        x = self.reduce_output(x.view(x.shape[0], x.shape[1]*x.shape[2]))
        p = self.policy_out(self.policy_activation(self.policy_layer(x)))
        v = self.value_out(self.value_activation(self.value_layer(x)))
        return p, v
