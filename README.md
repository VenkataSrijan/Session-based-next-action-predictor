# 🛒 Session-Based Next-Action Predictor

A deep learning system that predicts the next item a user will interact with
during an e-commerce browsing session, using only the sequence of items viewed
in the current session — no login, no history, no cookies required.

Built with PyTorch and deployed as an interactive Streamlit web application.

---

## 🎯 What This Does

Given a sequence of item interactions from a user session:
```
Item 133138 → Item 13576 → Item 8641 → Item 40870 → ?
```

The model predicts the top-K items the user is most likely to click next,
in real time.

---

## 📊 Models Implemented

| Model | Architecture | HR@5 | HR@10 | MRR@5 |
|---|---|---|---|---|
| GRU4Rec | Gated Recurrent Unit | 0.0044 | 0.0050 | 0.0022 |
| TransformerRec | Deep Transformer (4 heads, 3 layers) | 0.1674 | 0.1930 | 0.1302 |
| **SASRec** | **Self-Attentive Sequential** | **0.2042** | **0.2275** | **0.1514** |

> Evaluated on a held-out test set of 4,761 sequences (80/20 train-test split).
> SASRec achieves **46x better Hit Rate** than the GRU4Rec baseline.

---

## 🗂️ Project Structure
```
PR project/
├── app.py                    # Streamlit web application
├── evaluate_model.py         # Evaluation script (HR@K, MRR@K)
├── verify_predictions.py     # Prediction verification script
├── train_gru.py              # GRU4Rec training script
├── train_sasrec.py           # SASRec training script
├── train_transformer.py      # TransformerRec training script
├── requirements.txt
├── data/
│   └── events.csv            # RetailRocket dataset
└── src/
    ├── model.py              # GRURec architecture
    ├── sasrec.py             # SASRec architecture
    ├── transformer_rec.py    # TransformerRec architecture
    ├── preprocess.py         # Data loading and preprocessing
    ├── evaluate.py           # HR@K and MRR@K functions
    ├── predict.py            # Inference utilities
    ├── model.pth             # Trained GRU4Rec weights
    ├── sasrec_model.pth      # Trained SASRec weights
    ├── transformer_model.pth # Trained TransformerRec weights
    └── label_encoder.pkl     # Fitted item label encoder
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/session-next-action-predictor.git
cd session-next-action-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the RetailRocket dataset from Kaggle:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Place `events.csv` inside the `data/` folder.

### 4. Train the models
```bash
# Train GRU4Rec
python train_gru.py

# Train SASRec
python train_sasrec.py

# Train TransformerRec
python train_transformer.py
```

> For faster training, use Google Colab with a T4 GPU.
> Training on CPU is possible but slow.

### 5. Evaluate
```bash
python evaluate_model.py
```

### 6. Launch the app
```bash
python -m streamlit run app.py
```

---

## 🧠 How It Works

### Data Pipeline
- Raw RetailRocket clickstream events are grouped by visitor ID into sessions
- Sessions with fewer than 2 interactions are discarded
- Sliding window generates input-target pairs from each session
- Items are encoded into integer indices using a fitted LabelEncoder
- 80/20 chronological train-test split prevents data leakage

### Model Architectures

**GRU4Rec** — Embeds items into 64-dimensional vectors, processes them
through a 2-layer GRU with hidden size 128, and projects the final hidden
state to item logits. Serves as the recurrent baseline.

**SASRec** — Uses learnable item and positional embeddings fed into 2
Transformer encoder layers with 2 attention heads and causal masking.
Self-attention enables direct comparison between any two positions in the
session, identifying which past items are most predictive of the next interaction.

**TransformerRec** — A deeper variant with 4 attention heads and 3 encoder
layers. Additional LayerNorm is applied at both input and output. Requires
more data to fully realise its capacity advantage over SASRec.

### Training
- Loss: Cross-entropy over the full item vocabulary
- Optimiser: Adam (lr=0.001)
- Scheduler: StepLR (decay by 0.5 every 3 epochs)
- Epochs: 10
- Hardware: NVIDIA Tesla T4 GPU (Google Colab)

---

## 📈 Results

SASRec correctly predicts the next item in its top-5 recommendations
**20.42% of the time** on completely unseen sessions — from a catalogue
of 23,293 items. A random recommender would achieve approximately 0.021%.
```
Model           HR@5    HR@10   HR@20   MRR@5   MRR@10
─────────────────────────────────────────────────────
GRU4Rec         0.0044  0.0050  0.0086  0.0022  0.0023
SASRec          0.2042  0.2275  0.2506  0.1514  0.1545  ← best
TransformerRec  0.1674  0.1930  0.2153  0.1302  0.1334
```

---

## 🖥️ Demo App

The Streamlit app allows you to:
- Switch between all 3 models in real time
- Enter any sequence of item IDs as a session
- Get ranked top-K next-item predictions instantly
- View live model performance comparison

---

## 📦 Dataset

**RetailRocket E-Commerce Dataset**
- 2,756,101 interaction events
- 406,020 unique visitor sessions
- 235,061 unique items
- Event types: view, addtocart, transaction
- Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

---

## 📄 References

- Hidasi et al. (2016) — GRU4Rec: Session-based Recommendations with Recurrent Neural Networks
- Kang & McAuley (2018) — SASRec: Self-Attentive Sequential Recommendation
- Vaswani et al. (2017) — Attention Is All You Need
- Sun et al. (2019) — BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations

---

## 👥 Authors

- **Venkata Srijan** — Manipal Institute of Technology, Bengaluru
- **Sai Ashvith Suryadevara** — Manipal Institute of Technology, Bengaluru
- **Nara Bharath Chandra** — Manipal Institute of Technology, Bengaluru

---

## 📝 License

This project is for academic purposes.
