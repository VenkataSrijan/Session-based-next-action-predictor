import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pickle
from src.preprocess import load_data, create_sessions, filter_sessions, create_sequences, encode_items
from src.sasrec import SASRec
from src.predict import predict_next

print("Loading data...")
df = load_data("data/events.csv")
sessions = filter_sessions(create_sessions(df))
inputs, targets = create_sequences(sessions)
inputs  = inputs[:10000]
targets = targets[:10000]
inputs_encoded, targets_encoded, le_fresh = encode_items(inputs, targets)

with open("src/sasrec_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

model = SASRec(num_items=len(le.classes_))
model.load_state_dict(torch.load("src/sasrec_model.pth", map_location="cpu"))
model.eval()

known_classes = set(le.classes_.tolist())

# ── Scan through sequences and collect first 10 HITS ─────────────────────────
print("Scanning for confirmed hits...\n")

hits_found = []
checked = 0

for inp, tgt in zip(inputs, targets):
    if tgt not in known_classes:
        continue
    valid_input = [item for item in inp if item in known_classes]
    if not valid_input:
        continue

    preds = predict_next(model, valid_input, le, k=5)
    checked += 1

    if tgt in preds:
        hits_found.append((valid_input, tgt, list(preds)))

    if len(hits_found) == 10:
        break

print(f"{'='*65}")
print(f"  CONFIRMED HITS — First 10 sequences where model was correct")
print(f"  (Scanned {checked} sequences to find these)")
print(f"{'='*65}\n")

for i, (input_seq, actual_next, preds) in enumerate(hits_found):
    rank = [int(p) for p in preds].index(actual_next) + 1
    print(f"Hit #{i+1}:  ✅ Correct at rank #{rank}")
    print(f"  Input (last 3) : {input_seq[-3:]}")
    print(f"  Actual next    : {actual_next}")
    print(f"  Top-5 predicted: {[int(p) for p in preds]}")
    print()

print(f"{'='*65}")
print(f"  Found {len(hits_found)} hits in {checked} sequences checked")
print(f"  Observed HR@5 = {len(hits_found)/checked:.2%}  (expected ~6.67%)")
print(f"{'='*65}")