from torch.utils.data import Dataset
import chess

class SelfPlayDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (board, move_probs, outcome) = self.data[idx]
        moves_vec = torch.zeros(4096 + 88 + 2, dtype=torch.float)
        for move, prob in move_probs:
            moves_vec[move_to_index(board, move)] = prob
        return torch.tensor(tokenize_board(board)), move_vec, torch.tensor(outcome)

def move_to_index(board, move):
    if board.is_kingside_castling(move):
        return 4096 + 88 + 0
    elif board.is_queenside_castling(move):
        return 4096 + 88 + 1
    elif move.promotion:
        return 4096 + (move.promotion - 2)*22 + 3*(move.from_square % 8) + (-1 if move.to_square % 8 < move.from_square % 8 else (1 if move.to_square % 8 > move.from_square % 8 else 0))
    else:
        return 64*move.from_square + move.to_square

def tokenize_board(board):
    tokenized_board = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        piece_token = 0

        if piece:
            piece_token = piece.piece_type if piece.color == board.turn else piece.piece_type + 6
            if piece.piece_type == chess.KING and (
                    board.has_kingside_castling_rights(piece.color) or board.has_queenside_castling_rights(piece.color)):
                piece_token = 15 if piece.color == board.turn else 18
            elif piece.piece_type == chess.ROOK:
                if (piece.color == board.turn and square == chess.H1 and board.has_kingside_castling_rights(piece.color)) or \
                   (piece.color == board.turn and square == chess.A1 and board.has_queenside_castling_rights(piece.color)) or \
                   (piece.color != board.turn and square == chess.H8 and board.has_kingside_castling_rights(piece.color)) or \
                   (piece.color != board.turn and square == chess.A8 and board.has_queenside_castling_rights(piece.color)):
                    piece_token = 14 if piece.color == board.turn else 17
            elif board.ep_square:
                if square == board.ep_square:
                    piece_token = 13 if piece.color == board.turn else 16

        tokenized_board.append(piece_token)
    return tokenized_board

