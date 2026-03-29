import torch
import torch.nn as nn
import math


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec).
    Uses multi-head self-attention to capture long-range dependencies
    in user interaction sequences.
    """

    def __init__(self, num_items, embedding_dim=64, num_heads=2, num_layers=2,
                 max_seq_len=50, dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len   = max_seq_len

        # Item embedding + positional embedding
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, embedding_dim)

        # Stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True       # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(embedding_dim, num_items)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, T) — padded sequence of item indices
        B, T = x.shape

        # Clamp positions to max_seq_len
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        positions = positions.clamp(max=self.max_seq_len - 1)

        # Causal mask — each position can only attend to previous positions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Padding mask — ignore padded (0) positions
        pad_mask = (x == 0)   # (B, T)  True where padded

        x = self.dropout(self.item_emb(x) + self.pos_emb(positions))  # (B, T, D)
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        out = x[:, -1, :]     # Take last non-padding timestep → (B, D)
        return self.fc(out)   # (B, num_items)