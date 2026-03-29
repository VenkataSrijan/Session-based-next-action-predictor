import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from preprocess import load_data, create_sessions, create_sequences, encode_items, filter_sessions
from model import GRURec


# ── Dataset & collate must be at module level for Windows multiprocessing ──────
class SessionDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = [torch.tensor(s, dtype=torch.long) for s in sequences]
        self.targets   = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(batch):
    seqs, targets = zip(*batch)
    sorted_pairs  = sorted(zip(seqs, targets), key=lambda x: len(x[0]), reverse=True)
    seqs_sorted, targets_sorted = zip(*sorted_pairs)
    padded = pad_sequence(seqs_sorted, batch_first=True)
    return padded, torch.stack(targets_sorted)


# ── REQUIRED on Windows: all runtime code must be inside this guard ────────────
if __name__ == '__main__':

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data loading ──────────────────────────────────────────────────────────
  # ── Data loading ──────────────────────────────────────────────────────────
    print("Loading CSV...")
    df = load_data("../data/events.csv")
    print(f"Loaded {len(df)} rows")

    print("Creating sessions...")
    sessions = filter_sessions(create_sessions(df))
    print(f"Created {len(sessions)} sessions")

    print("Creating sequences...")
    inputs, targets = create_sequences(sessions)
    print(f"Created {len(inputs)} sequences — slicing to 50k")

    inputs  = inputs[:10000]
    targets = targets[:10000]

    print("Encoding items...")
    inputs_encoded, targets_encoded, le = encode_items(inputs, targets)
    print(f"Encoded. Vocab size: {len(le.classes_)}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = SessionDataset(inputs_encoded, targets_encoded)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,            # ← change to 0 (main process only, no spawning)
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=False, # ← must be False when num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = GRURec(num_items=len(le.classes_)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(batch_X)
            loss   = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(loader):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), "model.pth")

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Model and label encoder saved.")