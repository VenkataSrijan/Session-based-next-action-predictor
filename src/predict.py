import torch


def predict_next(model, session, le, k=5):
    """
    Predict the top-k next items given a session (list of raw item IDs).

    FIX 1: vectorized le.transform(session) instead of per-item loop
    FIX 2: torch.no_grad() to avoid building an unnecessary compute graph
    """
    model.eval()

    # FIX 1: single vectorized call instead of [le.transform([item])[0] for item in session]
    session_encoded = le.transform(session)
    seq = torch.tensor(session_encoded, dtype=torch.long).unsqueeze(0)  # (1, T)

    # FIX 2: no_grad was missing — was allocating a compute graph on every prediction
    with torch.no_grad():
        output = model(seq)                             # (1, num_items)

    top_k = torch.topk(output, k).indices[0]           # (k,)
    return le.inverse_transform(top_k.numpy())