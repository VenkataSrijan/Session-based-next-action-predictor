import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import pickle
from src.preprocess import load_data, create_sessions, filter_sessions, create_sequences, encode_items
from src.model import GRURec
from src.sasrec import SASRec
from src.transformer_rec import TransformerRec
from src.evaluate import hit_rate, mrr

# ── Load & prepare data ───────────────────────────────────────────────────────
print("Loading data...")
df = load_data("data/events.csv")
sessions = filter_sessions(create_sessions(df))
inputs, targets = create_sequences(sessions)
inputs  = inputs[:10000]
targets = targets[:10000]
inputs_encoded, targets_encoded, le = encode_items(inputs, targets)
print(f"Evaluation set: {len(inputs_encoded)} sequences | Vocab: {len(le.classes_)} items\n")

# ── Helper ────────────────────────────────────────────────────────────────────
results = {}

def evaluate_model(model, name):
    print(f"{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    results[name] = {}
    for k in [5, 10, 20]:
        hr    = hit_rate(model, inputs_encoded, targets_encoded, k=k)
        mrr_s = mrr(model, inputs_encoded, targets_encoded, k=k)
        results[name][k] = (hr, mrr_s)
        print(f"  HR@{k:<2} = {hr:.4f}   |   MRR@{k:<2} = {mrr_s:.4f}")
    print()

# ── GRU4Rec ───────────────────────────────────────────────────────────────────
print("Loading GRU4Rec...")
with open("src/label_encoder.pkl", "rb") as f:
    le_gru = pickle.load(f)
gru_model = GRURec(num_items=len(le_gru.classes_))
gru_model.load_state_dict(torch.load("src/model.pth", map_location="cpu"))
gru_model.eval()
evaluate_model(gru_model, "GRU4Rec")

# ── SASRec ────────────────────────────────────────────────────────────────────
print("Loading SASRec...")
with open("src/sasrec_label_encoder.pkl", "rb") as f:
    le_sas = pickle.load(f)
sas_model = SASRec(num_items=len(le_sas.classes_))
sas_model.load_state_dict(torch.load("src/sasrec_model.pth", map_location="cpu"))
sas_model.eval()
evaluate_model(sas_model, "SASRec")

# ── TransformerRec ────────────────────────────────────────────────────────────
print("Loading TransformerRec...")
with open("src/transformer_label_encoder.pkl", "rb") as f:
    le_tr = pickle.load(f)
tr_model = TransformerRec(num_items=len(le_tr.classes_))
tr_model.load_state_dict(torch.load("src/transformer_model.pth", map_location="cpu"))
tr_model.eval()
evaluate_model(tr_model, "TransformerRec")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"{'='*65}")
print(f"  SUMMARY — HR@10 and MRR@10 comparison")
print(f"{'='*65}")
print(f"  {'Model':<18} {'HR@5':>8} {'HR@10':>8} {'HR@20':>8} {'MRR@5':>8} {'MRR@10':>8}")
print(f"  {'-'*60}")
for name, scores in results.items():
    print(f"  {name:<18} {scores[5][0]:>8.4f} {scores[10][0]:>8.4f} {scores[20][0]:>8.4f} {scores[5][1]:>8.4f} {scores[10][1]:>8.4f}")
print(f"{'='*65}")
print("\nEvaluation complete!")