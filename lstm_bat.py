"""
lstm_bat.py  —  Battery Sensor BiLSTM+Attention Model
=======================================================
Features (9):
  Raw      : Volt, VoltR, Curr, CurrTot, Temp, Res
  Derived  : SoC (State-of-Charge proxy), Power (V×I), VoltDrop (VoltR-Volt)
Upgrade    : BiLSTM + Self-Attention, train/val split, gradient clipping,
             cosine LR schedule.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────
SEQ_LEN    = 15
EPOCHS     = 30
BATCH_SIZE = 64
MODEL_PATH = "bat_bilstm.pt"

# ── Load & Engineer Features ──────────────────────────────────
bat = pd.read_csv("BAT.csv")

df = pd.DataFrame()
for col in ["Volt", "VoltR", "Curr", "CurrTot", "Temp", "Res"]:
    df[col] = bat[col]

# State-of-Charge (linear proxy from voltage)
v_min, v_max = bat["Volt"].min(), bat["Volt"].max()
df["SoC"]      = (bat["Volt"] - v_min) / (v_max - v_min + 1e-8)
# Instantaneous power
df["Power"]    = bat["Volt"] * bat["Curr"]
# Voltage drop (reference − actual)
df["VoltDrop"] = bat["VoltR"] - bat["Volt"]
df = df.fillna(0)

INPUT_DIM = len(df.columns)
print(f"BAT features ({INPUT_DIM}): {df.columns.tolist()}")

# ── Scale & Sequence ──────────────────────────────────────────
scaler   = MinMaxScaler()
features = scaler.fit_transform(df.values.astype(np.float32))

X, y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i + SEQ_LEN])
    y.append(features[i + SEQ_LEN])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Train / Val split (85 / 15)
split = int(0.85 * len(X))
X_tr, X_val = X[:split], X[split:]
y_tr, y_val = y[:split], y[split:]

train_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                      batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                      batch_size=BATCH_SIZE)


# ── BiLSTM + Self-Attention ───────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h * 2, 1)

    def forward(self, out):
        w = F.softmax(self.attn(out), dim=1)
        return (w * out).sum(dim=1), w.squeeze(-1)


class BiLSTM_BAT(nn.Module):
    def __init__(self, in_dim, h=64, layers=2, drop=0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, h, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=drop if layers > 1 else 0.0)
        self.attn = SelfAttention(h)
        self.head = nn.Sequential(
            nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, in_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, aw = self.attn(out)
        return self.head(ctx), aw


# ── Train ─────────────────────────────────────────────────────
model     = BiLSTM_BAT(INPUT_DIM)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = float("inf")
print(f"\nTraining BAT BiLSTM+Attention ({EPOCHS} epochs) ...")
for epoch in range(EPOCHS):
    model.train()
    tr_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            pred, _ = model(xb)
            val_loss += criterion(pred, yb).item()

    scheduler.step()
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | train={tr_loss:.5f}  val={val_loss:.5f}"
          + (" ✅ best" if val_loss == best_val else ""))

# ── Evaluate ──────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

X_t = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    pred_last, attn_w = model(X_t[-1].unsqueeze(0))
    prediction = pred_last.numpy()[0]
    actual     = y[-1]

mae    = float(np.mean(np.abs(prediction - actual)))
per_ch = np.abs(prediction - actual)

print("\n─── BAT RESULT (BiLSTM+Attention) ───")
feature_names = df.columns.tolist()
print(f"{'Feature':<10} {'Predicted':>10} {'Actual':>10} {'Error':>8}")
print("─" * 42)
for fname, p, a, e in zip(feature_names, prediction, actual, per_ch):
    flag = " ◀ HIGH" if e > 0.15 else ""
    print(f"{fname:<10} {p:>10.4f} {a:>10.4f} {e:>8.4f}{flag}")

print(f"\nMean Absolute Error : {mae:.4f}")
print(f"Peak attention step : {attn_w.squeeze().argmax().item()} / {SEQ_LEN - 1}")

# Estimated SoC from prediction
soc_idx    = feature_names.index("SoC")
soc_pred   = prediction[soc_idx]
soc_actual = actual[soc_idx]
print(f"Estimated SoC (pred / actual): {soc_pred:.3f} / {soc_actual:.3f}")

if mae < 0.05:
    print("✅ BATTERY HEALTHY — all parameters nominal")
elif mae < 0.15:
    print("⚠️  BATTERY DEGRADATION — monitor voltage & temperature closely")
else:
    print("🚨 CRITICAL BATTERY ISSUE — power system stress detected")

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")