import torch
import torch.nn as nn


class TransformerRec(nn.Module):
    """
    Transformer-based Sequential Recommendation model.
    Differs from SASRec by using:
    - More attention heads (4 vs 2)
    - Deeper stack (3 layers vs 2)
    - Learnable positional embeddings with layer norm on input
    - Additional LayerNorm before final projection
    """

    def __init__(self, num_items, embedding_dim=64, num_heads=4, num_layers=3,
                 max_seq_len=50, dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len   = max_seq_len

        # Embeddings
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(embedding_dim, num_items)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        positions = positions.clamp(max=self.max_seq_len - 1)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Padding mask
        pad_mask = (x == 0)

        # Input: item + position embeddings with layer norm
        emb = self.input_norm(self.item_emb(x) + self.pos_emb(positions))
        emb = self.dropout(emb)

        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
        out = self.output_norm(out[:, -1, :])   # last timestep + norm
        out = self.dropout(out)
        return self.fc(out)                      # (B, num_items)