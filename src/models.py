"""Two-stage sequence model and a from-scratch ablation in PyTorch."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sequence_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        outputs, last = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True,
                                         total_length=x.shape[1])
        return outputs, last[-1]


class Autoencoder(nn.Module):
    """Variable-length GRU encoder and compressed-vector GRU decoder."""
    def __init__(self, input_size: int, hidden: int = 32):
        super().__init__()
        self.encoder = Encoder(input_size, hidden)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, input_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        _, latent = self.encoder(x, lengths)
        repeated = latent.unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)

    def errors(self, x: torch.Tensor, lengths: torch.Tensor):
        reconstruction = self(x, lengths)
        token_error = (reconstruction - x).square().mean(dim=-1)
        mask = sequence_mask(lengths, x.shape[1])
        score = (token_error * mask).sum(dim=1) / lengths.clamp_min(1)
        return score, token_error * mask


class AttentionClassifier(nn.Module):
    """Attention scores locate influential transactions; they are not causal proof."""
    def __init__(self, input_size: int, hidden: int = 32):
        super().__init__()
        self.encoder = Encoder(input_size, hidden)
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        states, _ = self.encoder(x, lengths)
        logits = self.attention(states).squeeze(-1)
        logits = logits.masked_fill(~sequence_mask(lengths, x.shape[1]), -1e9)
        weights = torch.softmax(logits, dim=1)
        context = torch.sum(states * weights.unsqueeze(-1), dim=1)
        return self.head(context).squeeze(-1), weights


def transfer_encoder(source: Autoencoder, target: AttentionClassifier):
    target.encoder.load_state_dict(source.encoder.state_dict())
