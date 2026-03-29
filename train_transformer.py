import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from src.preprocess import load_data, create_sessions, create_sequences, encode_items, filter_sessions
from src.transformer_rec import TransformerRec


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


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading CSV...")
    df = load_data("data/events.csv")
    print(f"Loaded {len(df)} rows")

    print("Creating sessions...")
    sessions = filter_sessions(create_sessions(df))
    print(f"Created {len(sessions)} sessions")

    print("Creating sequences...")
    inputs, targets = create_sequences(sessions)
    inputs  = inputs[:10000]
    targets = targets[:10000]
    print(f"Using {len(inputs)} sequences")

    print("Encoding items...")
    inputs_encoded, targets_encoded, le = encode_items(inputs, targets)
    print(f"Vocab size: {len(le.classes_)}")

    dataset = SessionDataset(inputs_encoded, targets_encoded)
    loader  = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model     = TransformerRec(num_items=len(le.classes_)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining TransformerRec...")
    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(batch_X)
            loss   = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "src/transformer_model.pth")
    with open("src/transformer_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nTransformerRec model and label encoder saved.")