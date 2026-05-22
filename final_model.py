"""
=============================================================================
  Intelligent Edge Computing for UAV Predictive Maintenance — Real Time
  S.P.E.C.T.R.A | PHM System v2.0
=============================================================================
  Architecture : Bidirectional LSTM + Self-Attention
  Sensors      : IMU (Acc + Gyr + derived), ATT (angles + tracking error),
                 BAT (Volt + Curr + SoC + Power)
  Features     : Fault injection, Per-sensor Health Index (EMA), Agentic
                 decision layer, Vibration trend, ZCR, CSV logging,
                 Rich ANSI dashboard
=============================================================================
"""

import os
import csv
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN           = 15       # longer temporal context
FAILURE_THRESHOLD = 0.40     # HI level that triggers RUL=0
EMA_ALPHA         = 0.08     # smoothing factor for health index (low = smoother)
PERSISTENCE       = 5        # steps for status persistence filter
MIN_DEGRADATION   = 0.0005   # noise floor to avoid infinite RUL
EPOCHS            = 15       # training epochs
BATCH_SIZE        = 64
FAULT_INJECT      = True     # toggle simulated fault injection
LOG_FILE          = "maintenance_log.csv"
PRINT_EVERY       = None     # auto-set per run length

# Per-sensor weight in composite score (must sum to 1)
SENSOR_WEIGHTS = {"imu": 0.45, "att": 0.30, "bat": 0.25}


# ─────────────────────────────────────────────
#  ANSI COLORS  (zero extra dependencies)
# ─────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"


# ─────────────────────────────────────────────
#  MODEL: BiLSTM + SELF-ATTENTION
# ─────────────────────────────────────────────
class SelfAttention(nn.Module):
    """Additive self-attention over LSTM time steps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        scores  = self.attn(lstm_out)              # (batch, seq, 1)
        weights = F.softmax(scores, dim=1)         # (batch, seq, 1)
        context = (weights * lstm_out).sum(dim=1)  # (batch, hidden*2)
        return context, weights.squeeze(-1)


class BiLSTM_Attention(nn.Module):
    """Two-layer Bidirectional LSTM with Self-Attention head."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=None, num_layers=2, dropout=0.25):
        super().__init__()
        output_dim = output_dim or input_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = SelfAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_w = self.attention(lstm_out)
        return self.head(context), attn_w


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_imu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand IMU features:
      • Raw: GyrX, GyrY, GyrZ, AccX, AccY, AccZ
      • Derived: vibration magnitude, gyro magnitude, rolling std (vibration
        trend), zero-crossing rate (ZCR) — all edge-computable
    """
    out = pd.DataFrame()
    out["GyrX"]   = df["GyrX"]
    out["GyrY"]   = df["GyrY"]
    out["GyrZ"]   = df["GyrZ"]
    out["AccX"]   = df["AccX"]
    out["AccY"]   = df["AccY"]
    out["AccZ"]   = df["AccZ"]

    # Vibration magnitude  √(Ax²+Ay²+Az²)
    out["VibMag"] = np.sqrt(df["AccX"]**2 + df["AccY"]**2 + df["AccZ"]**2)
    # Gyro magnitude
    out["GyrMag"] = np.sqrt(df["GyrX"]**2 + df["GyrY"]**2 + df["GyrZ"]**2)
    # Rolling std of vibration magnitude (vibration trend indicator)
    out["AccStd"] = out["VibMag"].rolling(5, min_periods=1).std().fillna(0)
    # Zero-Crossing Rate of AccX (edge-computable frequency proxy)
    out["ZCR"]    = (np.sign(df["AccX"]).diff().abs() > 0).astype(float)

    return out.fillna(0)


def engineer_att_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand ATT features:
      • Raw: DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw, ErrRP, ErrYaw
      • Derived: per-axis tracking errors (desired − actual)
    """
    out = pd.DataFrame()
    for col in ["DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw", "ErrRP", "ErrYaw"]:
        out[col] = df[col]

    out["RollErr"]  = df["DesRoll"]  - df["Roll"]
    out["PitchErr"] = df["DesPitch"] - df["Pitch"]
    out["YawErr"]   = df["DesYaw"]   - df["Yaw"]

    return out.fillna(0)


def engineer_bat_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand BAT features:
      • Raw: Volt, VoltR, Curr, CurrTot, Temp, Res
      • Derived: State-of-Charge (SoC) proxy, instantaneous power, voltage drop
    """
    out = pd.DataFrame()
    for col in ["Volt", "VoltR", "Curr", "CurrTot", "Temp", "Res"]:
        out[col] = df[col]

    v_min, v_max = df["Volt"].min(), df["Volt"].max()
    out["SoC"]      = (df["Volt"] - v_min) / (v_max - v_min + 1e-8)
    out["Power"]    = df["Volt"] * df["Curr"]
    out["VoltDrop"] = df["VoltR"] - df["Volt"]

    return out.fillna(0)


# ─────────────────────────────────────────────
#  FAULT INJECTION / SIMULATION
# ─────────────────────────────────────────────
def inject_motor_degradation(tensor: torch.Tensor, start_frac=0.70, severity=0.50):
    """
    Motor efficiency degradation: progressive amplitude growth on gyro channels
    (cols 0-2 in IMU features) from start_frac onwards.
    """
    n = tensor.size(0)
    start = int(n * start_frac)
    result = tensor.clone()
    for i in range(start, n):
        factor = 1 + severity * ((i - start) / (n - start))
        result[i, :, :3] = torch.clamp(result[i, :, :3] * factor, 0, 1)
    return result


def inject_propeller_imbalance(tensor: torch.Tensor, freq=0.25, amplitude=0.18):
    """
    Propeller imbalance: sinusoidal vibration added to AccX/AccY channels.
    """
    n = tensor.size(0)
    result = tensor.clone()
    t = torch.arange(n, dtype=torch.float32)
    noise = amplitude * torch.sin(2 * math.pi * freq * t)
    result[:, :, 3] = torch.clamp(result[:, :, 3] + noise.unsqueeze(1), 0, 1)
    result[:, :, 4] = torch.clamp(result[:, :, 4] + noise.unsqueeze(1) * 0.6, 0, 1)
    return result


def inject_power_stress(tensor: torch.Tensor, start_frac=0.75, volt_drop=0.28, curr_spike=0.35):
    """
    Power system stress: voltage drop + current spike in last fraction.
    """
    n = tensor.size(0)
    start = int(n * start_frac)
    result = tensor.clone()
    result[start:, :, 0] = torch.clamp(result[start:, :, 0] - volt_drop, 0, 1)  # Volt
    result[start:, :, 2] = torch.clamp(result[start:, :, 2] + curr_spike, 0, 1)  # Curr
    return result


# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_sensor(df_raw: pd.DataFrame, engineer_fn):
    """Scale → engineer features → create sliding-window sequences."""
    df       = engineer_fn(df_raw)
    cols     = df.columns.tolist()
    raw      = df.values.astype(np.float32)
    scaler   = MinMaxScaler()
    features = scaler.fit_transform(raw)

    X, y = [], []
    for i in range(len(features) - SEQ_LEN):
        X.append(features[i : i + SEQ_LEN])
        y.append(features[i + SEQ_LEN])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        scaler,
        cols,
    )


# ─────────────────────────────────────────────
#  TRAIN / LOAD
# ─────────────────────────────────────────────
def _train(X, y, input_dim, model_path):
    """Train BiLSTM+Attention with AdamW + cosine LR + gradient clipping."""
    split    = int(0.85 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE
    )

    model     = BiLSTM_Attention(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")
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
            torch.save(model.state_dict(), model_path)

        print(f"    Epoch {epoch+1:2d}/{EPOCHS} | "
              f"train={tr_loss:.5f}  val={val_loss:.5f}")

    return model


def load_or_train(X, y, input_dim, model_path, name):
    model = BiLSTM_Attention(input_dim)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"  {C.GREEN}✅ Loaded cached model → {model_path}{C.RESET}")
            model.eval()
            return model
        except Exception:
            pass

    print(f"\n  {C.CYAN}🚀 Training {name} BiLSTM+Attention ...{C.RESET}")
    _train(X, y, input_dim, model_path)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# ─────────────────────────────────────────────
#  HEALTH INDEX TRACKER (EMA)
# ─────────────────────────────────────────────
class HealthTracker:
    """
    Per-sensor Health Index using Exponential Moving Average.
    HI > 1.0 means deviation exceeds the baseline level.
    """
    def __init__(self, alpha=EMA_ALPHA, name="?"):
        self.alpha    = alpha
        self.name     = name
        self.ema_hi   = None
        self.baseline = None
        self._raw     = []
        self.history  = []

    def update(self, mae: float) -> float:
        self._raw.append(mae)
        # Use 50-sample median baseline → robust, normalises HI to ≈1.0 for clean data
        if self.baseline is None and len(self._raw) >= 50:
            self.baseline = float(np.median(self._raw[:50]))

        # Normalised HI: 1.0 = healthy baseline, >1.0 = degraded
        if self.baseline and self.baseline > 1e-8:
            hi = mae / self.baseline
        else:
            hi = 1.0          # hold neutral until baseline is ready
        hi = min(hi, 3.0)

        self.ema_hi = (
            hi if self.ema_hi is None
            else self.alpha * hi + (1 - self.alpha) * self.ema_hi
        )
        self.history.append(self.ema_hi)
        return self.ema_hi

    def degradation_rate(self, window=15) -> float:
        """Average HI change per step over last `window` steps."""
        if len(self.history) < window + 1:
            return 0.0
        return (self.history[-1] - self.history[-window]) / window

    def compute_rul(self) -> float:
        dr = self.degradation_rate()
        hi = self.ema_hi or 1.0
        # Threshold relative to normalised HI: 1.6 = 60% above baseline
        if dr > MIN_DEGRADATION:
            return max(0.0, (1.60 - hi) / dr)
        return float("inf")


# ─────────────────────────────────────────────
#  PER-SENSOR STATUS
# ─────────────────────────────────────────────
def sensor_status(tracker: HealthTracker) -> str:
    """
    Status based on normalised HI (1.0 = healthy baseline).
    Pre-baseline: always NORMAL so early steps don't false-alarm.
    """
    if tracker.baseline is None:
        return "NORMAL"          # not enough data yet
    hi = tracker.ema_hi or 1.0
    dr = tracker.degradation_rate()
    if hi > 1.55 or (hi > 1.35 and dr > 0.006):
        return "CRITICAL"
    elif hi > 1.20 or (hi > 1.10 and dr > 0.003):
        return "WARNING"
    return "NORMAL"


# ─────────────────────────────────────────────
#  AGENTIC DECISION LAYER
# ─────────────────────────────────────────────
_FAULT_MAP = {
    "imu": {
        "CRITICAL": ("Motor bearing failure imminent",
                     "Ground UAV immediately. Inspect all motor bearings."),
        "WARNING":  ("Elevated vibration trend detected",
                     "Schedule motor & propeller inspection within 10 cycles."),
        "NORMAL":   ("IMU nominal", "No action required."),
    },
    "att": {
        "CRITICAL": ("Severe propeller imbalance / flight instability",
                     "Replace propellers before next flight. Re-calibrate ESCs."),
        "WARNING":  ("Mild attitude tracking error growing",
                     "Check propeller balance and autopilot gains."),
        "NORMAL":   ("Attitude stable", "No action required."),
    },
    "bat": {
        "CRITICAL": ("Battery critical — power system stress detected",
                     "Land immediately. Replace battery pack. Check power rails."),
        "WARNING":  ("Battery degradation detected",
                     "Reduce payload. Schedule battery replacement within 5 cycles."),
        "NORMAL":   ("Battery healthy", "No action required."),
    },
}

_SEV = {"CRITICAL": 3, "WARNING": 2, "NORMAL": 1}


def agentic_decision(states: dict, ruls: dict, trackers: dict = None) -> dict:
    """
    Evaluate each sensor, rank by (severity, degradation_rate) so the
    fastest-degrading sensor wins ties. Produce dynamic, context-aware
    fault descriptions based on live HI and DR values.
    """
    def _dr(sensor):
        return trackers[sensor].degradation_rate() if trackers else 0.0

    def _hi(sensor):
        return (trackers[sensor].ema_hi or 1.0) if trackers else 1.0

    # Sort: severity first, then degradation rate as tiebreaker
    ranked = sorted(
        states.items(),
        key=lambda kv: (_SEV[kv[1]], _dr(kv[0])),
        reverse=True,
    )
    top_sensor, top_state = ranked[0]
    base_fault, base_action = _FAULT_MAP[top_sensor][top_state]

    # ── Dynamic urgency prefix based on live DR ───────────────────
    dr  = _dr(top_sensor)
    hi  = _hi(top_sensor)
    rul = ruls.get(top_sensor, float("inf"))

    if top_state == "CRITICAL":
        if dr > 0.015:
            urgency     = "⛔ RAPID DEGRADATION — "
            rul_hint    = f"RUL ≈ {rul:.0f} steps" if rul != float("inf") else "RUL exhausted"
            fault_txt   = urgency + base_fault
            action_txt  = f"{base_action} {rul_hint}."
        elif dr > 0.006:
            urgency     = "🔴 ACCELERATING FAULT — "
            fault_txt   = urgency + base_fault
            action_txt  = base_action
        else:
            fault_txt   = base_fault + f" (HI={hi:.3f})"
            action_txt  = base_action
    elif top_state == "WARNING":
        if dr > 0.004:
            fault_txt   = f"⚠ Trending to critical — {base_fault}"
            action_txt  = f"URGENT: {base_action}"
        else:
            fault_txt   = base_fault + f" (HI={hi:.3f}, DR={dr:+.4f})"
            action_txt  = base_action
    else:  # NORMAL
        # Even in NORMAL, flag which sensor has highest DR
        worst_dr_sensor = max(states.keys(), key=_dr)
        wdr = _dr(worst_dr_sensor)
        if wdr > 0.001:
            fault_txt  = f"All nominal — monitor {worst_dr_sensor.upper()} (rising trend)"
            action_txt = f"Routine check on {worst_dr_sensor.upper()} recommended."
        else:
            fault_txt  = base_fault
            action_txt = base_action

    # ── Confidence: fraction of sensors at same severity ──────────
    top_sev    = _SEV[top_state]
    confidence = sum(1 for _, s in ranked if _SEV[s] == top_sev) / len(ranked)

    # ── Multi-sensor summary ──────────────────────────────────────
    fastest_sensor = max(states.keys(), key=_dr)

    return {
        "priority_sensor":  top_sensor.upper(),
        "top_state":        top_state,
        "fault":            fault_txt,
        "action":           action_txt,
        "confidence":       confidence,
        "fastest_sensor":   fastest_sensor.upper(),
        "fastest_dr":       round(_dr(fastest_sensor), 5),
        "per_sensor":       {k: (_FAULT_MAP[k][v][0], ruls[k]) for k, v in states.items()},
    }


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────
def _status_color(s: str) -> str:
    if s == "CRITICAL":  return f"{C.BOLD}{C.RED}CRITICAL 🚨{C.RESET}"
    if s == "WARNING":   return f"{C.BOLD}{C.YELLOW}WARNING  ⚠️ {C.RESET}"
    return f"{C.BOLD}{C.GREEN}NORMAL   ✅{C.RESET}"


def _rul_bar(rul, width=28) -> str:
    if rul == float("inf"):
        bar   = "█" * width
        label = "STABLE       "
        color = C.GREEN
    else:
        filled = min(int((rul / 150) * width), width)
        bar    = "█" * filled + "░" * (width - filled)
        label  = f"RUL {rul:6.1f} steps"
        color  = C.GREEN if filled > width * 0.6 else (C.YELLOW if filled > width * 0.25 else C.RED)
    return f"{color}[{bar}] {label}{C.RESET}"


def _hi_mini(hi, width=12) -> str:
    hi     = min(hi, 1.0)
    filled = int(hi * width)
    bar    = "▮" * filled + "▯" * (width - filled)
    color  = C.RED if hi > 0.65 else (C.YELLOW if hi > 0.35 else C.GREEN)
    return f"{color}{bar}{C.RESET}"


def print_dashboard(t, trackers, states, ruls, agent, step):
    W = 66
    print(f"\n{C.BOLD}{C.BLUE}{'═' * W}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  ✈  UAV PHM SYSTEM  │  Step {t:5d}  │  {datetime.now().strftime('%H:%M:%S')}{C.RESET}")
    print(f"{C.BOLD}{C.BLUE}{'═' * W}{C.RESET}")

    # Per-sensor rows
    hdr = f"  {'Sensor':6s} │ {'Health Index':12s} │ {'HI':5s} │ {'Trend':>8s} │ {'MAE':>6s} │ Status"
    print(f"{C.DIM}{hdr}{C.RESET}")
    print(f"{C.DIM}  {'─' * (W - 2)}{C.RESET}")

    for name, tracker in trackers.items():
        hi  = tracker.ema_hi or 0.0
        dr  = tracker.degradation_rate()
        mae = step[f"{name}_mae"]
        st  = states[name]
        arrow = (f"{C.RED}↑↑" if dr > 0.003 else
                 f"{C.YELLOW}↑ " if dr > 0.001 else
                 f"{C.GREEN}→ " if abs(dr) <= 0.001 else
                 f"{C.GREEN}↓ ") + C.RESET
        print(f"  {name.upper():6s} │ {_hi_mini(hi)} │ {hi:5.3f} │ {arrow}{dr:+.4f} │ {mae:6.4f} │ {_status_color(st)}")

    print()
    # Composite RUL bar
    worst_rul = min(ruls.values())
    print(f"  {_rul_bar(worst_rul)}")
    print()

    # Agentic decision box
    conf_str = f"{int(agent['confidence'] * 100)}%"
    print(f"  {C.MAGENTA}{C.BOLD}⚡ AGENTIC DECISION LAYER  [confidence: {conf_str}]{C.RESET}")
    print(f"  {C.BOLD}Priority Sensor :{C.RESET} {C.CYAN}{agent['priority_sensor']}{C.RESET}")
    print(f"  {C.BOLD}Detected Fault  :{C.RESET} {agent['fault']}")
    print(f"  {C.BOLD}Recommended Act :{C.RESET} {C.YELLOW}{agent['action']}{C.RESET}")

    if FAULT_INJECT:
        print(f"  {C.DIM}[Fault injection active — motor degradation + power stress]{C.RESET}")


# ─────────────────────────────────────────────
#  CSV LOGGER
# ─────────────────────────────────────────────
def init_log():
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "step",
            "imu_mae", "att_mae", "bat_mae",
            "imu_hi",  "att_hi",  "bat_hi",
            "imu_dr",  "att_dr",  "bat_dr",
            "imu_status", "att_status", "bat_status",
            "imu_rul",    "att_rul",    "bat_rul",
            "system_status", "fault", "action", "agent_confidence",
        ])


def log_step(t, step, trackers, states, ruls, agent, sys_status):
    def fmt(v):
        return "STABLE" if v == float("inf") else f"{v:.5f}"

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(), t,
            f"{step['imu_mae']:.5f}", f"{step['att_mae']:.5f}", f"{step['bat_mae']:.5f}",
            f"{(trackers['imu'].ema_hi or 0):.5f}",
            f"{(trackers['att'].ema_hi or 0):.5f}",
            f"{(trackers['bat'].ema_hi or 0):.5f}",
            f"{trackers['imu'].degradation_rate():.6f}",
            f"{trackers['att'].degradation_rate():.6f}",
            f"{trackers['bat'].degradation_rate():.6f}",
            states["imu"], states["att"], states["bat"],
            fmt(ruls["imu"]), fmt(ruls["att"]), fmt(ruls["bat"]),
            sys_status,
            agent["fault"], agent["action"],
            f"{agent['confidence']:.2f}",
        ])


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global PRINT_EVERY

    # ── Load raw CSVs ──────────────────────────────────────────
    print(f"\n{C.BOLD}{C.CYAN}{'─'*50}")
    print(f"  ✈  S.P.E.C.T.R.A — UAV Predictive Health Monitoring v2.0")
    print(f"{'─'*50}{C.RESET}")
    print(f"\n{C.CYAN}[ 1/4 ] Loading sensor CSVs ...{C.RESET}")

    imu_raw = pd.read_csv("IMU.csv")
    att_raw = pd.read_csv("ATT.csv")
    bat_raw = pd.read_csv("BAT.csv")

    # ── Feature Engineering ────────────────────────────────────
    print(f"{C.CYAN}[ 2/4 ] Engineering features ...{C.RESET}")

    imu_X, imu_y, _, imu_cols = prepare_sensor(imu_raw, engineer_imu_features)
    att_X, att_y, _, att_cols = prepare_sensor(att_raw, engineer_att_features)
    bat_X, bat_y, _, bat_cols = prepare_sensor(bat_raw, engineer_bat_features)

    print(f"  IMU  → {len(imu_cols)} features: {imu_cols}")
    print(f"  ATT  → {len(att_cols)} features: {att_cols}")
    print(f"  BAT  → {len(bat_cols)} features: {bat_cols}")

    # ── Load / Train Models ────────────────────────────────────
    print(f"\n{C.CYAN}[ 3/4 ] Loading / Training models ...{C.RESET}")

    imu_model = load_or_train(imu_X, imu_y, len(imu_cols), "imu_bilstm.pt", "IMU")
    att_model = load_or_train(att_X, att_y, len(att_cols), "att_bilstm.pt", "ATT")
    bat_model = load_or_train(bat_X, bat_y, len(bat_cols), "bat_bilstm.pt", "BAT")

    # Convert to tensors
    imu_Xt = torch.tensor(imu_X, dtype=torch.float32)
    att_Xt = torch.tensor(att_X, dtype=torch.float32)
    bat_Xt = torch.tensor(bat_X, dtype=torch.float32)

    # ── Fault Injection (on inference sequences) ───────────────
    if FAULT_INJECT:
        print(f"\n  {C.YELLOW}⚡ Injecting simulated faults ...{C.RESET}")
        imu_Xt = inject_motor_degradation(imu_Xt,    start_frac=0.70, severity=0.50)
        imu_Xt = inject_propeller_imbalance(imu_Xt,  freq=0.25, amplitude=0.18)
        bat_Xt = inject_power_stress(bat_Xt,          start_frac=0.75)
        print(f"    Motor degradation  → IMU  (from 70% of flight)")
        print(f"    Propeller imbalance→ IMU  (sinusoidal, full flight)")
        print(f"    Power system stress→ BAT  (from 75% of flight)")

    # ── PHM Loop ───────────────────────────────────────────────
    print(f"\n{C.CYAN}[ 4/4 ] Running PHM inference loop ...{C.RESET}\n")

    trackers = {
        "imu": HealthTracker(name="imu"),
        "att": HealthTracker(name="att"),
        "bat": HealthTracker(name="bat"),
    }
    status_history = []
    init_log()

    min_len    = min(len(imu_Xt), len(att_Xt), len(bat_Xt))
    PRINT_EVERY = max(1, min_len // 20)   # ~20 dashboard prints per run

    for t in range(min_len):

        # ── Inference ─────────────────────────────────────────
        with torch.no_grad():
            imu_pred, _ = imu_model(imu_Xt[t].unsqueeze(0))
            att_pred, _ = att_model(att_Xt[t].unsqueeze(0))
            bat_pred, _ = bat_model(bat_Xt[t].unsqueeze(0))

        imu_mae = float(np.mean(np.abs(imu_pred.numpy()[0] - imu_y[t])))
        att_mae = float(np.mean(np.abs(att_pred.numpy()[0] - att_y[t])))
        bat_mae = float(np.mean(np.abs(bat_pred.numpy()[0] - bat_y[t])))

        # ── Health Index Update (EMA) ──────────────────────────
        trackers["imu"].update(imu_mae)
        trackers["att"].update(att_mae)
        trackers["bat"].update(bat_mae)

        # ── Per-sensor Status ──────────────────────────────────
        states = {k: sensor_status(trackers[k]) for k in trackers}

        # ── Per-sensor RUL ─────────────────────────────────────
        ruls = {k: trackers[k].compute_rul() for k in trackers}

        # ── System-level Status (worst-case) ──────────────────
        max_sev = max(_SEV[s] for s in states.values())
        sys_raw = {3: "CRITICAL 🚨", 2: "WARNING ⚠️", 1: "NORMAL ✅"}[max_sev]
        status_history.append(sys_raw)

        # Persistence filter (majority vote)
        if len(status_history) >= PERSISTENCE:
            recent  = status_history[-PERSISTENCE:]
            sys_out = max(set(recent), key=recent.count)
        else:
            sys_out = sys_raw

        # ── Agentic Decision ───────────────────────────────────
        agent = agentic_decision(states, ruls)

        step_data = {
            "imu_mae": imu_mae, "att_mae": att_mae, "bat_mae": bat_mae,
        }

        # ── Log ────────────────────────────────────────────────
        log_step(t, step_data, trackers, states, ruls, agent, sys_out)

        # ── Display ────────────────────────────────────────────
        if t % PRINT_EVERY == 0 or "CRITICAL" in sys_out:
            print_dashboard(t, trackers, states, ruls, agent, step_data)

    print(f"\n{C.BOLD}{C.GREEN}✅  PHM run complete!{C.RESET}")
    print(f"   Processed {min_len} time steps.")
    print(f"   Log saved → {C.CYAN}{LOG_FILE}{C.RESET}\n")


if __name__ == "__main__":
    main()