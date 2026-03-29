import torch
from torch.nn.utils.rnn import pad_sequence


def hit_rate(model, inputs, targets, k=5, batch_size=512):
    """
    Compute Hit Rate @ k over all (input, target) pairs.

    FIX: batched inference instead of one forward pass per sample
    Old code ran len(inputs) separate forward passes → extremely slow
    """
    model.eval()
    hits = 0

    seqs = [torch.tensor(s, dtype=torch.long) for s in inputs]
    tgts = torch.tensor(targets, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            batch_tgts = tgts[i : i + batch_size]

            padded = pad_sequence(batch_seqs, batch_first=True)
            output = model(padded)                          # (B, num_items)

            top_k = torch.topk(output, k).indices          # (B, k)
            match = (top_k == batch_tgts.unsqueeze(1)).any(dim=1)  # (B,)
            hits += match.sum().item()

    return hits / len(inputs)


def mrr(model, inputs, targets, k=5, batch_size=512):
    """
    Compute Mean Reciprocal Rank @ k over all (input, target) pairs.

    FIX: batched inference instead of one forward pass per sample
    Old code ran len(inputs) separate forward passes → extremely slow
    """
    model.eval()
    score = 0.0

    seqs = [torch.tensor(s, dtype=torch.long) for s in inputs]
    tgts = torch.tensor(targets, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            batch_tgts = tgts[i : i + batch_size]

            padded = pad_sequence(batch_seqs, batch_first=True)
            output = model(padded)                          # (B, num_items)

            top_k = torch.topk(output, k).indices          # (B, k)

            for j, tgt in enumerate(batch_tgts):
                match = (top_k[j] == tgt).nonzero(as_tuple=True)[0]
                if len(match) > 0:
                    rank   = match[0].item() + 1           # 1-indexed
                    score += 1.0 / rank

    return score / len(inputs)