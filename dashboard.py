"""
dashboard.py — UAV PHM Live Monitoring Dashboard
=================================================
Streams real-time PHM inference results to a premium web dashboard
via Flask + Server-Sent Events (SSE). Models load from cached .pt files.

Usage : python3 dashboard.py
Opens : http://localhost:5000
"""

import os, sys, json, time, threading, webbrowser
import numpy as np
import pandas as pd
import torch
from queue import Queue, Empty

# ── Auto-install Flask if missing ─────────────────────────────────────────────
try:
    from flask import Flask, Response
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "-q"])
    from flask import Flask, Response

# ── Import PHM engine from final_model.py ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_model import (
    BiLSTM_Attention,
    engineer_imu_features, engineer_att_features, engineer_bat_features,
    prepare_sensor, load_or_train,
    inject_motor_degradation, inject_power_stress,
    HealthTracker, sensor_status, agentic_decision, _FAULT_MAP, _SEV,
    SEQ_LEN, MIN_DEGRADATION, EPOCHS, BATCH_SIZE,
)

# ── Config ────────────────────────────────────────────────────────────────────
STEP_DELAY = 0.55   # seconds per step  (206 steps ≈ 113 s per cycle — smooth demo)
REPLAY     = True   # loop indefinitely
PORT       = 8050

# ── Broadcast state ───────────────────────────────────────────────────────────
_clients_lock : threading.Lock = threading.Lock()
_clients      : list           = []

def broadcast(data: dict):
    with _clients_lock:
        dead = []
        for q in _clients:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            _clients.remove(q)

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return Response(HTML_TEMPLATE, mimetype="text/html")

@app.route("/stream")
def stream():
    q: Queue = Queue(maxsize=600)
    with _clients_lock:
        _clients.append(q)

    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

# ── PHM Inference Thread ──────────────────────────────────────────────────────
def run_inference():
    print("\n  [ PHM Engine ] Loading sensor data ...")
    imu_raw = pd.read_csv("IMU.csv")
    att_raw = pd.read_csv("ATT.csv")
    bat_raw = pd.read_csv("BAT.csv")

    imu_X, imu_y, _, imu_cols = prepare_sensor(imu_raw, engineer_imu_features)
    att_X, att_y, _, att_cols = prepare_sensor(att_raw, engineer_att_features)
    bat_X, bat_y, _, bat_cols = prepare_sensor(bat_raw, engineer_bat_features)

    print("  [ PHM Engine ] Loading BiLSTM+Attention models ...")
    imu_model = load_or_train(imu_X, imu_y, len(imu_cols), "imu_bilstm.pt", "IMU")
    att_model = load_or_train(att_X, att_y, len(att_cols), "att_bilstm.pt", "ATT")
    bat_model = load_or_train(bat_X, bat_y, len(bat_cols), "bat_bilstm.pt", "BAT")

    imu_Xt = torch.tensor(imu_X, dtype=torch.float32)
    att_Xt = torch.tensor(att_X, dtype=torch.float32)
    bat_Xt = torch.tensor(bat_X, dtype=torch.float32)

    # Fault injection on inference sequences
    imu_Xt = inject_motor_degradation(imu_Xt, start_frac=0.70, severity=0.50)
    bat_Xt = inject_power_stress(bat_Xt, start_frac=0.75)

    min_len = min(len(imu_Xt), len(att_Xt), len(bat_Xt))
    print(f"  [ PHM Engine ] Ready — streaming {min_len} steps at {STEP_DELAY}s/step")
    print(f"  [ PHM Engine ] Dashboard → http://localhost:{PORT}\n")

    # Wait for browser to connect
    time.sleep(2.0)

    while True:
        trackers = {
            "imu": HealthTracker(name="imu"),
            "att": HealthTracker(name="att"),
            "bat": HealthTracker(name="bat"),
        }
        status_history = []
        PERSIST = 5

        broadcast({"restart": True, "total": min_len})
        time.sleep(0.5)

        for t in range(min_len):
            with torch.no_grad():
                imu_pred, _ = imu_model(imu_Xt[t].unsqueeze(0))
                att_pred, _ = att_model(att_Xt[t].unsqueeze(0))
                bat_pred, _ = bat_model(bat_Xt[t].unsqueeze(0))

            imu_mae = float(np.mean(np.abs(imu_pred.numpy()[0] - imu_y[t])))
            att_mae = float(np.mean(np.abs(att_pred.numpy()[0] - att_y[t])))
            bat_mae = float(np.mean(np.abs(bat_pred.numpy()[0] - bat_y[t])))

            trackers["imu"].update(imu_mae)
            trackers["att"].update(att_mae)
            trackers["bat"].update(bat_mae)

            states  = {k: sensor_status(trackers[k]) for k in trackers}
            ruls    = {k: trackers[k].compute_rul()  for k in trackers}
            max_sev = max(_SEV[s] for s in states.values())
            sys_status = {3: "CRITICAL", 2: "WARNING", 1: "NORMAL"}[max_sev]
            status_history.append(sys_status)

            if len(status_history) >= PERSIST:
                recent  = status_history[-PERSIST:]
                sys_out = max(set(recent), key=recent.count)
            else:
                sys_out = sys_status

            agent    = agentic_decision(states, ruls, trackers)
            worst_rul = min(ruls.values())
            rul_val  = float(worst_rul) if worst_rul != float("inf") else "STABLE"

            broadcast({
                "step": t, "total": min_len,
                "system_status": sys_out,

                "imu_hi":     round(float(trackers["imu"].ema_hi or 1.0), 5),
                "imu_status": states["imu"],
                "imu_mae":    round(imu_mae, 5),
                "imu_dr":     round(float(trackers["imu"].degradation_rate()), 6),
                "imu_fault":  _FAULT_MAP["imu"][states["imu"]][0],

                "att_hi":     round(float(trackers["att"].ema_hi or 1.0), 5),
                "att_status": states["att"],
                "att_mae":    round(att_mae, 5),
                "att_dr":     round(float(trackers["att"].degradation_rate()), 6),
                "att_fault":  _FAULT_MAP["att"][states["att"]][0],

                "bat_hi":     round(float(trackers["bat"].ema_hi or 1.0), 5),
                "bat_status": states["bat"],
                "bat_mae":    round(bat_mae, 5),
                "bat_dr":     round(float(trackers["bat"].degradation_rate()), 6),
                "bat_fault":  _FAULT_MAP["bat"][states["bat"]][0],

                "rul": rul_val,

                "agent_sensor":     agent["priority_sensor"],
                "agent_fault":      agent["fault"],
                "agent_action":     agent["action"],
                "agent_confidence": round(agent["confidence"], 2),
                "fastest_sensor":   agent["fastest_sensor"],
                "fastest_dr":       agent["fastest_dr"],
                "system_trend":     (
                    "DEGRADING" if any(trackers[k].degradation_rate() > 0.003 for k in trackers)
                    else "STABLE"
                ),
            })

            time.sleep(STEP_DELAY)

        if not REPLAY:
            broadcast({"done": True})
            break
        time.sleep(3.0)   # pause before replay


# ── HTML Dashboard ────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S.P.E.C.T.R.A · UAV Predictive Health Monitoring</title>
<meta name="description" content="S.P.E.C.T.R.A — Sensor Prognostics & Edge Computing for Telemetry-based Real-time Analysis. BiLSTM+Attention PHM for autonomous UAV systems.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:        #040a12;
  --bg2:       #060e1a;
  --glass:     rgba(255,255,255,0.035);
  --gborder:   rgba(255,255,255,0.07);
  --normal:    #00e676;
  --warning:   #ffb300;
  --critical:  #ff1744;
  --imu:       #00b8d4;
  --att:       #9c6eff;
  --bat:       #ffab00;
  --text:      rgba(255,255,255,0.92);
  --dim:       rgba(255,255,255,0.42);
  --muted:     rgba(255,255,255,0.20);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Background grid + scan line ────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,184,212,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,184,212,.04) 1px, transparent 1px);
  background-size: 44px 44px;
  pointer-events: none;
}
body::after {
  content: '';
  position: fixed; left: 0; width: 100%; height: 1px; z-index: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(0,184,212,.5) 50%, transparent 100%);
  animation: scanLine 6s linear infinite;
  pointer-events: none;
}
@keyframes scanLine { 0%{top:-2px} 100%{top:100vh} }

/* ── Layout ─────────────────────────────────────────────── */
.wrap { position: relative; z-index: 1; max-width: 1380px; margin: 0 auto; padding: 18px 20px; }

/* ── Header ─────────────────────────────────────────────── */
.header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 22px;
  background: rgba(0,184,212,.07);
  border: 1px solid rgba(0,184,212,.18);
  border-radius: 14px; margin-bottom: 18px;
  backdrop-filter: blur(12px);
}
.h-left { display: flex; align-items: center; gap: 14px; }
.uav-icon { font-size: 30px; animation: drift 3s ease-in-out infinite; }
@keyframes drift { 0%,100%{transform:translateY(0) rotate(0deg)} 50%{transform:translateY(-5px) rotate(2deg)} }
.h-title { font-family:'Orbitron',monospace; font-size:18px; font-weight:700; color:var(--imu); letter-spacing:2px; }
.h-sub   { font-size:10px; color:var(--dim); letter-spacing:1.5px; margin-top:3px; text-transform:uppercase; }
.h-right { display:flex; align-items:center; gap:22px; }

.step-block { text-align:center; }
.step-val   { font-family:'Orbitron',monospace; font-size:26px; font-weight:900; color:var(--imu); }
.step-lbl   { font-size:9px; color:var(--dim); letter-spacing:2px; text-transform:uppercase; margin-bottom:4px; }
.prog-wrap  { height:3px; background:rgba(255,255,255,0.08); border-radius:2px; width:90px; margin:0 auto; overflow:hidden; }
.prog-fill  { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--imu),var(--att)); transition:width .8s; }

.sys-badge {
  padding:9px 20px; border-radius:9px;
  font-family:'Orbitron',monospace; font-size:13px; font-weight:700; letter-spacing:2px;
  transition:all 1.2s;
}
.sys-NORMAL   { background:rgba(0,230,118,.12); border:1px solid var(--normal); color:var(--normal); }
.sys-WARNING  { background:rgba(255,179,0,.12);  border:1px solid var(--warning); color:var(--warning); animation:pulse-w 1.6s ease-in-out infinite; }
.sys-CRITICAL { background:rgba(255,23,68,.12);  border:1px solid var(--critical); color:var(--critical); animation:pulse-c .9s ease-in-out infinite; }
@keyframes pulse-w { 0%,100%{box-shadow:0 0 8px rgba(255,179,0,.2)} 50%{box-shadow:0 0 24px rgba(255,179,0,.7)} }
@keyframes pulse-c { 0%,100%{box-shadow:0 0 10px rgba(255,23,68,.3)} 50%{box-shadow:0 0 32px rgba(255,23,68,.9)} }

/* ── Sensor cards ───────────────────────────────────────── */
.sensor-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:18px; }
.s-card {
  background:var(--glass); border:1px solid var(--gborder);
  border-radius:16px; padding:22px; backdrop-filter:blur(10px);
  transition:border-color 1.4s, box-shadow 1.4s;
}
.s-card.c-NORMAL   { border-color:rgba(0,230,118,.18); }
.s-card.c-WARNING  { border-color:rgba(255,179,0,.4); box-shadow:0 0 22px rgba(255,179,0,.08); }
.s-card.c-CRITICAL { border-color:rgba(255,23,68,.5); box-shadow:0 0 32px rgba(255,23,68,.14); }

.s-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }
.s-name { font-family:'Orbitron',monospace; font-size:15px; font-weight:700; letter-spacing:3px; }
.s-name.imu { color:var(--imu); }
.s-name.att { color:var(--att); }
.s-name.bat { color:var(--bat); }

.s-badge { padding:4px 10px; border-radius:6px; font-family:'Orbitron',monospace; font-size:10px; font-weight:700; letter-spacing:1px; transition:all 1.2s; }
.b-NORMAL   { background:rgba(0,230,118,.12); color:var(--normal); border:1px solid rgba(0,230,118,.3); }
.b-WARNING  { background:rgba(255,179,0,.12);  color:var(--warning); border:1px solid rgba(255,179,0,.4); animation:pulse-w 1.6s infinite; }
.b-CRITICAL { background:rgba(255,23,68,.12);  color:var(--critical); border:1px solid rgba(255,23,68,.4); animation:pulse-c .9s infinite; }

.gauge-wrap { display:flex; justify-content:center; margin:8px 0 16px; }

.metrics { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.m-box { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.05); border-radius:9px; padding:10px 12px; }
.m-lbl { font-size:9px; color:var(--dim); letter-spacing:1px; text-transform:uppercase; margin-bottom:5px; }
.m-val { font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:500; }

.fault-strip {
  margin-top:12px; padding:9px 12px; border-radius:8px;
  font-size:11px; border-left:2px solid transparent;
  background:rgba(255,255,255,.02);
  transition:all 1.0s;
}
.fs-NORMAL   { border-left-color:var(--normal);   color:rgba(0,230,118,.75); }
.fs-WARNING  { border-left-color:var(--warning);  color:rgba(255,179,0,.85); }
.fs-CRITICAL { border-left-color:var(--critical); color:rgba(255,50,80,.9); }

/* ── RUL bar ─────────────────────────────────────────────── */
.panel { background:var(--glass); border:1px solid var(--gborder); border-radius:16px; padding:20px 24px; margin-bottom:18px; backdrop-filter:blur(10px); }
.panel-title {
  font-family:'Orbitron',monospace; font-size:11px; font-weight:700;
  color:var(--dim); letter-spacing:3px; text-transform:uppercase;
  margin-bottom:16px; display:flex; align-items:center; gap:10px;
}
.panel-title::before { content:''; display:block; width:3px; height:14px; background:var(--imu); border-radius:2px; }

.rul-track { position:relative; height:40px; background:rgba(255,255,255,.05); border-radius:10px; overflow:hidden; }
.rul-fill  { height:100%; border-radius:10px; transition:width 1.8s ease, background 1.8s; position:relative; overflow:hidden; }
.rul-fill::after {
  content:''; position:absolute; top:0; left:-120%; width:60%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.15),transparent);
  animation:shimmer 2.2s infinite;
}
@keyframes shimmer { 0%{left:-120%} 100%{left:200%} }
.rul-label {
  position:absolute; top:50%; right:16px; transform:translateY(-50%);
  font-family:'Orbitron',monospace; font-size:13px; font-weight:700;
}

/* ── Chart ───────────────────────────────────────────────── */
.chart-inner { position:relative; height:230px; }

/* ── Agent panel ─────────────────────────────────────────── */
.agent-panel {
  background:rgba(156,110,255,.05); border:1px solid rgba(156,110,255,.18);
  border-radius:16px; padding:20px 24px; margin-bottom:18px;
  backdrop-filter:blur(10px);
}
.agent-title {
  font-family:'Orbitron',monospace; font-size:11px; font-weight:700;
  color:var(--att); letter-spacing:3px;
  display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;
}
.conf-pill {
  font-family:'Orbitron',monospace; font-size:12px; font-weight:700; color:var(--att);
  background:rgba(156,110,255,.15); border:1px solid rgba(156,110,255,.3);
  padding:4px 12px; border-radius:6px;
}
.agent-rows { display:grid; grid-template-columns:110px 1fr; row-gap:10px; align-items:start; }
.a-key { font-size:10px; color:var(--dim); letter-spacing:1px; text-transform:uppercase; padding-top:3px; }
.a-val { font-size:14px; }
.a-sensor { font-family:'Orbitron',monospace; font-size:18px; font-weight:700; color:var(--imu); }
.a-fault  { color:var(--text); }
.a-action {
  padding:11px 16px; border-radius:9px; font-size:13px; font-weight:500;
  transition:all 1.2s;
}
.a-action.ac-NORMAL   { background:rgba(0,230,118,.08); border:1px solid rgba(0,230,118,.2); color:var(--normal); }
.a-action.ac-WARNING  { background:rgba(255,179,0,.08);  border:1px solid rgba(255,179,0,.25); color:var(--warning); }
.a-action.ac-CRITICAL { background:rgba(255,23,68,.08);  border:1px solid rgba(255,23,68,.3); color:var(--critical); }

/* ── Footer ─────────────────────────────────────────────── */
.footer { display:flex; align-items:center; justify-content:space-between; padding:10px 0; color:var(--muted); font-size:11px; }
.live-row { display:flex; align-items:center; gap:6px; }
.live-dot { width:7px; height:7px; border-radius:50%; background:var(--normal); animation:blink .8s infinite; }
.inj-pill { display:flex; align-items:center; gap:6px; color:rgba(255,179,0,.65); font-size:11px; }
.inj-dot  { width:6px; height:6px; border-radius:50%; background:var(--warning); animation:blink 1.1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── Boot overlay ───────────────────────────────────────── */
.boot {
  position:fixed; inset:0; background:var(--bg);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  z-index:200; transition:opacity .6s;
}
.boot-spinner {
  width:64px; height:64px; border-radius:50%;
  border:3px solid rgba(0,184,212,.15);
  border-top-color:var(--imu);
  animation:spin 1s linear infinite; margin-bottom:22px;
}
@keyframes spin { to{transform:rotate(360deg)} }
.boot-txt { font-family:'Orbitron',monospace; color:var(--imu); letter-spacing:4px; font-size:13px; }
.boot-sub { font-size:11px; color:var(--dim); margin-top:8px; letter-spacing:2px; }

/* ── Done toast ─────────────────────────────────────────── */
.toast {
  position:fixed; bottom:36px; right:36px; z-index:150;
  background:rgba(0,230,118,.12); border:1px solid var(--normal);
  border-radius:12px; padding:14px 24px;
  font-family:'Orbitron',monospace; color:var(--normal);
  font-size:12px; letter-spacing:2px;
  display:none; animation:fadeIn .4s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

/* ── Trend badge ─────────────────────────────────────────── */
.trend-badge {
  padding:6px 14px; border-radius:8px;
  font-family:'Orbitron',monospace; font-size:11px; font-weight:700; letter-spacing:2px;
  transition:all 1.0s;
}
.trend-STABLE    { background:rgba(0,230,118,.1); border:1px solid rgba(0,230,118,.25); color:var(--normal); }
.trend-DEGRADING { background:rgba(255,179,0,.1); border:1px solid rgba(255,179,0,.35); color:var(--warning); animation:pulse-w .9s infinite; }

/* ── Strip ───────────────────────────────────────────────── */
.strip-item {
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06);
  border-radius:9px; padding:10px 12px;
}
.strip-name { font-family:'Orbitron',monospace; font-size:11px; font-weight:700; letter-spacing:2px; margin-bottom:4px; }
.strip-name.imu{color:var(--imu)} .strip-name.att{color:var(--att)} .strip-name.bat{color:var(--bat)}
.strip-status { font-size:12px; font-weight:600; margin-bottom:3px; }
.strip-hint { font-size:10px; color:var(--dim); font-family:'JetBrains Mono',monospace; }

/* ── Responsive ─────────────────────────────────────────── */
@media(max-width:900px) {
  .sensor-grid { grid-template-columns:1fr; }
  .header { flex-direction:column; gap:14px; text-align:center; }
}
</style>
</head>
<body>

<div class="boot" id="boot">
  <div class="boot-spinner"></div>
  <div class="boot-txt">S.P.E.C.T.R.A · INITIALIZING</div>
  <div class="boot-sub">Sensor Prognostics & Edge Computing for Telemetry-based Real-time Analysis</div>
</div>

<div class="toast" id="toast">✅ CYCLE COMPLETE — REPLAYING</div>

<div class="wrap">

  <!-- ─── HEADER ─────────────────────────────────────────── -->
  <div class="header">
    <div class="h-left">
      <div class="uav-icon">✈</div>
      <div>
        <div class="h-title">S.P.E.C.T.R.A</div>
        <div class="h-sub">Sensor Prognostics & Edge Computing for Telemetry-based Real-time Analysis · BiLSTM+Attention</div>
      </div>
    </div>
    <div class="h-right">
      <div class="step-block">
        <div class="step-lbl">STEP</div>
        <div class="step-val" id="stepVal">—</div>
        <div class="prog-wrap"><div class="prog-fill" id="progFill" style="width:0%"></div></div>
      </div>
      <div id="trendBadge" class="trend-badge trend-STABLE">STABLE →</div>
      <div id="sysBadge" class="sys-badge sys-NORMAL">NORMAL ✅</div>
    </div>
  </div>

  <!-- ─── SENSOR CARDS ───────────────────────────────────── -->
  <div class="sensor-grid">

    <!-- IMU -->
    <div class="s-card c-NORMAL" id="imuCard">
      <div class="s-head">
        <div class="s-name imu">IMU</div>
        <div class="s-badge b-NORMAL" id="imuBadge">NORMAL</div>
      </div>
      <div class="gauge-wrap"><canvas id="imuGauge" width="160" height="155"></canvas></div>
      <div class="metrics">
        <div class="m-box">
          <div class="m-lbl">MAE Error</div>
          <div class="m-val" id="imuMAE" style="color:var(--imu)">—</div>
        </div>
        <div class="m-box">
          <div class="m-lbl">Degradation</div>
          <div class="m-val" id="imuDR">—</div>
        </div>
      </div>
      <div class="fault-strip fs-NORMAL" id="imuFault">IMU nominal</div>
    </div>

    <!-- ATT -->
    <div class="s-card c-NORMAL" id="attCard">
      <div class="s-head">
        <div class="s-name att">ATT</div>
        <div class="s-badge b-NORMAL" id="attBadge">NORMAL</div>
      </div>
      <div class="gauge-wrap"><canvas id="attGauge" width="160" height="155"></canvas></div>
      <div class="metrics">
        <div class="m-box">
          <div class="m-lbl">MAE Error</div>
          <div class="m-val" id="attMAE" style="color:var(--att)">—</div>
        </div>
        <div class="m-box">
          <div class="m-lbl">Degradation</div>
          <div class="m-val" id="attDR">—</div>
        </div>
      </div>
      <div class="fault-strip fs-NORMAL" id="attFault">Attitude stable</div>
    </div>

    <!-- BAT -->
    <div class="s-card c-NORMAL" id="batCard">
      <div class="s-head">
        <div class="s-name bat">BAT</div>
        <div class="s-badge b-NORMAL" id="batBadge">NORMAL</div>
      </div>
      <div class="gauge-wrap"><canvas id="batGauge" width="160" height="155"></canvas></div>
      <div class="metrics">
        <div class="m-box">
          <div class="m-lbl">MAE Error</div>
          <div class="m-val" id="batMAE" style="color:var(--bat)">—</div>
        </div>
        <div class="m-box">
          <div class="m-lbl">Degradation</div>
          <div class="m-val" id="batDR">—</div>
        </div>
      </div>
      <div class="fault-strip fs-NORMAL" id="batFault">Battery healthy</div>
    </div>

  </div><!-- /sensor-grid -->

  <!-- ─── RUL BAR ─────────────────────────────────────────── -->
  <div class="panel">
    <div class="panel-title">Remaining Useful Life — Composite (Worst-Case Sensor)</div>
    <div class="rul-track">
      <div class="rul-fill" id="rulFill" style="width:100%;background:linear-gradient(90deg,var(--normal),var(--imu));"></div>
      <div class="rul-label" id="rulLabel" style="color:var(--normal)">STABLE</div>
    </div>
  </div>

  <!-- ─── HEALTH CHART ─────────────────────────────────────── -->
  <div class="panel">
    <div class="panel-title">Health Index History — Real-Time (All Sensors)</div>
    <div class="chart-inner"><canvas id="hiChart"></canvas></div>
  </div>

  <!-- ─── AGENTIC PANEL ─────────────────────────────────────── -->
  <div class="agent-panel">
    <div class="agent-title">
      <span>⚡ AGENTIC DECISION LAYER</span>
      <span class="conf-pill" id="confPill">— CONF</span>
    </div>
    <div class="agent-rows">
      <div class="a-key">Priority Sensor</div>
      <div class="a-val a-sensor" id="agSensor">—</div>
      <div class="a-key">Fastest Degrading</div>
      <div class="a-val" id="agFastest" style="color:var(--att);font-family:'Orbitron',monospace;font-size:13px">—</div>
      <div class="a-key">Detected Fault</div>
      <div class="a-val a-fault"  id="agFault">Awaiting stream...</div>
      <div class="a-key">Recommended Action</div>
      <div class="a-val a-action ac-NORMAL" id="agAction">Awaiting stream...</div>
    </div>
    <!-- Per-sensor status strip -->
    <div style="margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.06)">
      <div style="font-size:9px;color:var(--dim);letter-spacing:2px;margin-bottom:10px">PER-SENSOR SNAPSHOT</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px" id="sensorStrip">
        <div class="strip-item" id="stripIMU">
          <div class="strip-name imu">IMU</div>
          <div class="strip-status" id="stripIMUStatus">NORMAL</div>
          <div class="strip-hint" id="stripIMUHint">—</div>
        </div>
        <div class="strip-item" id="stripATT">
          <div class="strip-name att">ATT</div>
          <div class="strip-status" id="stripATTStatus">NORMAL</div>
          <div class="strip-hint" id="stripATTHint">—</div>
        </div>
        <div class="strip-item" id="stripBAT">
          <div class="strip-name bat">BAT</div>
          <div class="strip-status" id="stripBATStatus">NORMAL</div>
          <div class="strip-hint" id="stripBATHint">—</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ─── FOOTER ───────────────────────────────────────────── -->
  <div class="footer">
    <div class="inj-pill">
      <span class="inj-dot"></span>
      Fault Injection: Motor Degradation (70%) · Power Stress (75%)
    </div>
    <div class="live-row">
      <span class="live-dot"></span>
      <span id="clock">--:--:--</span> &nbsp;·&nbsp; S.P.E.C.T.R.A v1.0 &nbsp;·&nbsp; BiLSTM+Attention
    </div>
  </div>

</div><!-- /wrap -->

<script>
// ── Color maps ─────────────────────────────────────────────────────────────
const SC = { NORMAL:'#00e676', WARNING:'#ffb300', CRITICAL:'#ff1744' };
const SENSORC = { imu:'#00b8d4', att:'#9c6eff', bat:'#ffab00' };

function sColor(s){ return SC[s] || '#00e676'; }

// ── Smooth gauge system (RAF-based lerp) ───────────────────────────────────
const gaugeTarget  = { imu:1.0, att:1.0, bat:1.0 };
const gaugeCurrent = { imu:1.0, att:1.0, bat:1.0 };
const gaugeStatus  = { imu:'NORMAL', att:'NORMAL', bat:'NORMAL' };
let   gaugeRAF = null;

function _drawGauge(id, hi, status, accent) {
  const cv = document.getElementById(id);
  if(!cv) return;
  const ctx = cv.getContext('2d');
  const W=cv.width, H=cv.height, cx=W/2, cy=H/2+8, r=56;
  ctx.clearRect(0,0,W,H);
  const SA=Math.PI*.75, EA=Math.PI*2.25, arc=EA-SA;

  // Track
  ctx.beginPath(); ctx.arc(cx,cy,r,SA,EA);
  ctx.strokeStyle='rgba(255,255,255,0.07)';
  ctx.lineWidth=13; ctx.lineCap='round'; ctx.stroke();

  // Value arc
  const pct=Math.min(hi/1.6,1);
  const col=sColor(status);
  if(pct>0.001){
    ctx.beginPath(); ctx.arc(cx,cy,r,SA,SA+pct*arc);
    ctx.strokeStyle=col; ctx.lineWidth=13; ctx.lineCap='round';
    ctx.shadowColor=col; ctx.shadowBlur=20; ctx.stroke(); ctx.shadowBlur=0;
  }

  // Threshold tick marks (WARNING=1.2, CRITICAL=1.55, full=1.6)
  [[1.2,'rgba(255,179,0,0.5)'],[1.55,'rgba(255,23,68,0.5)']].forEach(([v,c])=>{
    const p=v/1.6, a=SA+p*arc;
    const x1=cx+Math.cos(a)*(r-9), y1=cy+Math.sin(a)*(r-9);
    const x2=cx+Math.cos(a)*(r+5), y2=cy+Math.sin(a)*(r+5);
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
    ctx.strokeStyle=c; ctx.lineWidth=2; ctx.stroke();
  });

  // Center HI value
  ctx.fillStyle=col;
  ctx.font='bold 19px "Orbitron", monospace';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(hi.toFixed(3),cx,cy-7);
  ctx.fillStyle='rgba(255,255,255,0.30)';
  ctx.font='9px Inter,sans-serif';
  ctx.fillText('HEALTH INDEX',cx,cy+12);

  // Accent dot
  ctx.beginPath(); ctx.arc(cx,cy-r-14,4.5,0,Math.PI*2);
  ctx.fillStyle=accent; ctx.shadowColor=accent; ctx.shadowBlur=8;
  ctx.fill(); ctx.shadowBlur=0;
}

function _animateGauges() {
  let still = true;
  ['imu','att','bat'].forEach(s=>{
    const diff = gaugeTarget[s] - gaugeCurrent[s];
    if(Math.abs(diff) > 0.0003){
      gaugeCurrent[s] += diff * 0.06;   // lerp factor — lower = slower/smoother
      still = false;
    } else {
      gaugeCurrent[s] = gaugeTarget[s];
    }
    _drawGauge(s+'Gauge', gaugeCurrent[s], gaugeStatus[s], SENSORC[s]);
  });
  if(!still) gaugeRAF = requestAnimationFrame(_animateGauges);
  else        gaugeRAF = null;
}

function setGaugeTarget(sensor, hi, status) {
  gaugeTarget[sensor]  = hi;
  gaugeStatus[sensor]  = status;
  if(!gaugeRAF) gaugeRAF = requestAnimationFrame(_animateGauges);
}

// ── Chart.js ───────────────────────────────────────────────────────────────
const MAX_PTS=80;
const cdata={
  labels:[],
  datasets:[
    {label:'IMU',  data:[],borderColor:'#00b8d4',backgroundColor:'rgba(0,184,212,.08)',tension:.4,borderWidth:2,pointRadius:0,fill:true},
    {label:'ATT',  data:[],borderColor:'#9c6eff',backgroundColor:'rgba(156,110,255,.08)',tension:.4,borderWidth:2,pointRadius:0,fill:true},
    {label:'BAT',  data:[],borderColor:'#ffab00',backgroundColor:'rgba(255,171,0,.08)',tension:.4,borderWidth:2,pointRadius:0,fill:true},
  ]
};
const hiChart=new Chart(document.getElementById('hiChart'),{
  type:'line', data:cdata,
  options:{
    responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{
      legend:{labels:{color:'rgba(255,255,255,0.55)',font:{family:'Inter',size:12},boxWidth:12,usePointStyle:true}},
      tooltip:{
        backgroundColor:'rgba(4,10,18,.95)',titleColor:'#fff',bodyColor:'rgba(255,255,255,.7)',
        borderColor:'rgba(255,255,255,.1)',borderWidth:1
      }
    },
    scales:{
      x:{ticks:{color:'rgba(255,255,255,.25)',maxTicksLimit:10,font:{size:10}},grid:{color:'rgba(255,255,255,.04)'}},
      y:{min:0,ticks:{color:'rgba(255,255,255,.25)',font:{size:10}},grid:{color:'rgba(255,255,255,.04)'},
         title:{display:true,text:'Health Index (HI)',color:'rgba(255,255,255,.3)',font:{size:10}}}
    }
  }
});

// ── Update helpers ─────────────────────────────────────────────────────────
function updateCard(sensor, d){
  const hi=d[sensor+'_hi'], status=d[sensor+'_status'], mae=d[sensor+'_mae'], dr=d[sensor+'_dr'], fault=d[sensor+'_fault'];
  const col=sColor(status);

  document.getElementById(sensor+'Card').className='s-card c-'+status;
  const badge=document.getElementById(sensor+'Badge');
  badge.textContent=status; badge.className='s-badge b-'+status;

  // Smooth gauge via lerp instead of instant redraw
  setGaugeTarget(sensor, hi, status);

  const maeEl=document.getElementById(sensor+'MAE');
  maeEl.textContent=mae.toFixed(4); maeEl.style.color=col;

  const drEl=document.getElementById(sensor+'DR');
  const sign=dr>=0?'+':'';
  drEl.textContent=sign+dr.toFixed(5);
  drEl.style.color=dr>0.001?'#ff1744':(dr<-0.001?'#00e676':'rgba(255,255,255,0.65)');

  const faultEl=document.getElementById(sensor+'Fault');
  faultEl.textContent=fault; faultEl.className='fault-strip fs-'+status;
}

function updateRUL(rul){
  const fill=document.getElementById('rulFill'), label=document.getElementById('rulLabel');
  if(rul==='STABLE'){
    fill.style.width='100%';
    fill.style.background='linear-gradient(90deg,var(--normal),var(--imu))';
    label.textContent='STABLE'; label.style.color='var(--normal)';
  } else {
    const pct=Math.min((rul/150)*100,100);
    const col=rul<10?'#ff1744':(rul<35?'#ffb300':'#00e676');
    fill.style.width=pct+'%'; fill.style.background=col;
    label.textContent='RUL: '+rul.toFixed(1)+' steps'; label.style.color=col;
  }
}

function updateSystem(status){
  const badge=document.getElementById('sysBadge');
  const emoji={NORMAL:'✅',WARNING:'⚠️',CRITICAL:'🚨'};
  badge.textContent=status+' '+(emoji[status]||'');
  badge.className='sys-badge sys-'+status;
}

function updateAgent(d){
  document.getElementById('agSensor').textContent=d.agent_sensor;
  document.getElementById('agFault').textContent=d.agent_fault;
  const act=document.getElementById('agAction');
  act.textContent=d.agent_action; act.className='a-val a-action ac-'+d.system_status;
  document.getElementById('confPill').textContent=Math.round(d.agent_confidence*100)+'% CONF';

  // Fastest-degrading sensor
  const fdr=d.fastest_dr!==undefined?d.fastest_dr:0;
  const fsEl=document.getElementById('agFastest');
  if(fsEl) fsEl.textContent=d.fastest_sensor + (fdr>0?' (+'+fdr.toFixed(4)+'/step)':'');

  // Trend badge
  const tb=document.getElementById('trendBadge');
  if(tb){
    const tr=d.system_trend||'STABLE';
    tb.textContent=tr==='DEGRADING'?'DEGRADING ↑':'STABLE →';
    tb.className='trend-badge trend-'+tr;
  }

  // Per-sensor strip
  const sensors=['IMU','ATT','BAT'];
  const keys   =['imu','att','bat'];
  const emojis ={NORMAL:'✅',WARNING:'⚠️',CRITICAL:'🚨'};
  sensors.forEach((s,i)=>{
    const k=keys[i];
    const st=d[k+'_status'];
    const dr=d[k+'_dr'];
    const hi=d[k+'_hi'];
    const stEl=document.getElementById('strip'+s+'Status');
    const htEl=document.getElementById('strip'+s+'Hint');
    if(stEl){ stEl.textContent=(emojis[st]||'')+' '+st; stEl.style.color=sColor(st); }
    if(htEl){ htEl.textContent='HI='+hi.toFixed(3)+' DR='+(dr>=0?'+':'')+dr.toFixed(4); }
  });
}

function pushChart(step, ihi, ahi, bhi){
  cdata.labels.push(step);
  cdata.datasets[0].data.push(ihi);
  cdata.datasets[1].data.push(ahi);
  cdata.datasets[2].data.push(bhi);
  if(cdata.labels.length>MAX_PTS){
    cdata.labels.shift();
    cdata.datasets.forEach(ds=>ds.data.shift());
  }
  hiChart.update('none');
}

// ── Clock ──────────────────────────────────────────────────────────────────
setInterval(()=>{
  const n=new Date();
  document.getElementById('clock').textContent=n.toTimeString().slice(0,8);
},1000);

// ── Toast helper ───────────────────────────────────────────────────────────
function showToast(msg,dur=3000){
  const t=document.getElementById('toast');
  t.textContent=msg; t.style.display='block';
  setTimeout(()=>{ t.style.opacity='0'; setTimeout(()=>{ t.style.display='none'; t.style.opacity='1'; },600); },dur);
}

// ── SSE stream ─────────────────────────────────────────────────────────────
let firstData=true, totalSteps=206;

const src=new EventSource('/stream');

src.onmessage=function(e){
  const d=JSON.parse(e.data);

  if(firstData){
    document.getElementById('boot').style.opacity='0';
    setTimeout(()=>document.getElementById('boot').style.display='none',650);
    firstData=false;
  }

  if(d.restart){
    totalSteps=d.total||206;
    // Reset chart on replay
    cdata.labels=[];
    cdata.datasets.forEach(ds=>ds.data=[]);
    hiChart.update('none');
    return;
  }

  if(d.done){
    src.close();
    showToast('✅ ANALYSIS COMPLETE',6000);
    return;
  }

  // Step progress
  document.getElementById('stepVal').textContent=d.step;
  document.getElementById('progFill').style.width=((d.step/totalSteps)*100)+'%';

  // System
  updateSystem(d.system_status);

  // Sensors
  updateCard('imu',d);
  updateCard('att',d);
  updateCard('bat',d);

  // RUL
  updateRUL(d.rul);

  // Chart
  pushChart(d.step, d.imu_hi, d.att_hi, d.bat_hi);

  // Agent
  updateAgent(d);
};

src.onerror=function(){
  // Browser auto-reconnects SSE — no action needed
};
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Launch inference background thread
    t = threading.Thread(target=run_inference, daemon=True)
    t.start()

    # Auto-open browser after 1.8 s
    threading.Timer(1.8, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    print(f"\n{'─'*58}")
    print(f"  ✈  S.P.E.C.T.R.A — UAV Predictive Health Monitoring")
    print(f"{'─'*58}")
    print(f"  URL    → http://localhost:{PORT}")
    print(f"  Steps  → 206  |  Delay → {STEP_DELAY}s  (~{int(206*STEP_DELAY)}s/cycle)")
    print(f"  Models → BiLSTM+Attention (cached .pt files)")
    print(f"{'─'*58}\n")

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
