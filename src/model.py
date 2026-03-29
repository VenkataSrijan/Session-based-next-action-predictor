import torch
import torch.nn as nn


class GRURec(nn.Module):
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        # FIX: added num_layers and dropout for better regularization
        # dropout only applied between layers when num_layers > 1
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # FIX: added dropout before final projection to reduce overfitting
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, num_items)

    def forward(self, x):
        x        = self.embedding(x)           # (B, T, embedding_dim)
        out, _   = self.gru(x)                 # (B, T, hidden_dim)
        out      = out[:, -1, :]               # last timestep → (B, hidden_dim)
        out      = self.dropout(out)
        out      = self.fc(out)                # (B, num_items)
        return out