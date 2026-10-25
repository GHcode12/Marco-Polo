import serial
import re
import time
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from collections import deque

# ================= CONFIGURATION =================
SERIAL_PORT = "COM3"          # Your Receiver ESP32 port
WEB_PORT = 8080               # Opens http://localhost:8080
WINDOW_SIZE = 60              # 60 packets (~0.6s rolling window for motion)
BREATH_WINDOW = 800           # 800 packets (~8 seconds for high-precision BPM)
CALIBRATION_SECONDS = 6       # Empty-room calibration duration
VITALS_LOCK_SECONDS = 3       # Time to lock onto stationary subject's chest
# =================================================

radar_state = {
    "connected": False,
    "mode": "MOTION",             # "MOTION" or "BREATHING"
    "calibrating": True,
    "cal_remaining": CALIBRATION_SECONDS,
    "status": "INITIALIZING",
    "human_detected": False,
    "confidence_score": 0.0,
    "threshold": 3.2,
    "breathing_bpm": 0.0,
    "breathing_conf": 0,          # 0 to 100% SNR lock confidence
    "breath_wave": [0.0] * 150,   # Live inhale/exhale waveform for UI
    "subcarriers": [0.0] * 50,
    "history": [0.0] * 120,
    "trigger_recal": False,
    "trigger_vitals": False
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wi-Fi CSI Through-Wall Tactical & Vitals Radar</title>
<style>
  :root { --bg: #090d16; --panel: #111827; --border: #1f2937; --green: #10b981; --red: #ef4444; --amber: #f59e0b; --cyan: #06b6d4; }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
  body { background: var(--bg); color: #f3f4f6; padding: 20px; height: 100vh; display: flex; flex-direction: column; gap: 16px; overflow: hidden; }
  
  .top-bar { display: flex; gap: 14px; align-items: stretch; }
  .status-banner { flex: 1; background: var(--panel); border: 2px solid var(--border); border-radius: 12px; padding: 16px 22px; display: flex; align-items: center; justify-content: space-between; transition: all 0.2s ease; }
  .status-banner.clear { border-color: var(--green); box-shadow: 0 0 25px rgba(16, 185, 129, 0.15); }
  .status-banner.vitals { background: rgba(6, 182, 212, 0.12); border-color: var(--cyan); box-shadow: 0 0 30px rgba(6, 182, 212, 0.3); }
  .status-banner.alert { background: rgba(127, 29, 29, 0.35); border-color: var(--red); box-shadow: 0 0 35px rgba(239, 68, 68, 0.4); }
  .status-banner.cal { border-color: var(--amber); }
  .status-text { font-size: 25px; font-weight: 800; letter-spacing: 0.8px; }
  .pulse-dot { width: 18px; height: 18px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 12px var(--cyan); }

  .controls { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px 18px; display: flex; align-items: center; gap: 14px; }
  .btn { color: #fff; border: none; padding: 12px 16px; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 13px; transition: 0.15s; white-space: nowrap; }
  .btn-motion { background: #2563eb; }
  .btn-motion:hover { background: #1d4ed8; transform: translateY(-1px); }
  .btn-vitals { background: #0891b2; border: 1px solid #22d3ee; box-shadow: 0 0 15px rgba(6, 182, 212, 0.3); }
  .btn-vitals:hover { background: #0e7490; transform: translateY(-1px); }
  .slider-box { display: flex; flex-direction: column; gap: 4px; font-size: 11px; color: #9ca3af; }
  input[type=range] { width: 130px; accent-color: var(--cyan); cursor: pointer; }

  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; }
  .card-label { font-size: 11px; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
  .card-val { font-size: 30px; font-weight: 800; margin-top: 6px; font-family: 'Consolas', monospace; }

  .visuals { flex: 1; display: grid; grid-template-columns: 1fr 1.5fr; gap: 16px; min-height: 0; }
  .viz-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; }
  .viz-title { font-size: 13px; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; display: flex; justify-content: space-between; }
  canvas { flex: 1; width: 100%; height: 100%; border-radius: 8px; background: #06090f; }
</style>
</head>
<body>

<div class="top-bar">
  <div id="banner" class="status-banner cal">
    <div>
      <div id="modeTag" style="font-size:11px; color:#9ca3af; font-weight:700; letter-spacing:1.5px;">MODE: MOTION RADAR // COM3</div>
      <div id="statusText" class="status-text">CONNECTING TO RECEIVER...</div>
    </div>
    <div id="dot" class="pulse-dot"></div>
  </div>

  <div class="controls">
    <div class="slider-box">
      <span>SENSITIVITY (<b id="thVal" style="color:#fff">3.2</b>)</span>
      <input id="thSlider" type="range" min="1.2" max="8.0" step="0.2" value="3.2" oninput="updateThresh(this.value)">
    </div>
    <button class="btn btn-motion" onclick="recalibrate()">🔄 CALIBRATE MOTION (6s)</button>
    <button class="btn btn-vitals" onclick="startVitals()">🫁 SCAN VITALS / BREATHING</button>
  </div>
</div>

<div class="metrics">
  <div class="card">
    <div class="card-label">Motion Confidence (Z-Score)</div>
    <div id="scoreVal" class="card-val" style="color: var(--cyan);">0.00</div>
  </div>
  <div class="card">
    <div class="card-label">Respiration Rate</div>
    <div id="bpmVal" class="card-val" style="color: var(--green);">-- BPM</div>
  </div>
  <div class="card">
    <div class="card-label">Breathing Lock Quality</div>
    <div id="confVal" class="card-val" style="color: var(--amber);">0%</div>
  </div>
  <div class="card">
    <div class="card-label">Trigger Threshold</div>
    <div id="threshCard" class="card-val" style="color: #a855f7;">3.20</div>
  </div>
</div>

<div class="visuals">
  <div class="viz-panel">
    <div class="viz-title"><span id="leftTitle">Tactical Radar Scope</span><span>2.4 GHz CSI</span></div>
    <canvas id="radarCanvas"></canvas>
  </div>
  <div class="viz-panel">
    <div class="viz-title"><span id="rightTitle">Live Motion Timeline & 50-Subcarrier Spectrum</span><span>100 PKT/S</span></div>
    <canvas id="graphCanvas"></canvas>
  </div>
</div>

<script>
const radarCanvas = document.getElementById('radarCanvas');
const rCtx = radarCanvas.getContext('2d');
const graphCanvas = document.getElementById('graphCanvas');
const gCtx = graphCanvas.getContext('2d');
let sweepAngle = 0;
let latestData = null;

function resizeCanvases() {
  radarCanvas.width = radarCanvas.clientWidth;
  radarCanvas.height = radarCanvas.clientHeight;
  graphCanvas.width = graphCanvas.clientWidth;
  graphCanvas.height = graphCanvas.clientHeight;
}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();

function updateThresh(val) {
  document.getElementById('thVal').innerText = parseFloat(val).toFixed(1);
  document.getElementById('threshCard').innerText = parseFloat(val).toFixed(2);
  fetch('/set_threshold?val=' + val);
}

function recalibrate() { fetch('/recalibrate'); }
function startVitals() { fetch('/start_vitals'); }

async function pollState() {
  try {
    const res = await fetch('/state');
    latestData = await res.json();
    
    const banner = document.getElementById('banner');
    const stText = document.getElementById('statusText');
    const modeTag = document.getElementById('modeTag');
    const dot = document.getElementById('dot');

    document.getElementById('scoreVal').innerText = latestData.confidence_score.toFixed(2);
    document.getElementById('confVal').innerText = latestData.breathing_conf + "%";
    document.getElementById('bpmVal').innerText = latestData.breathing_bpm > 0 ? latestData.breathing_bpm.toFixed(1) + " BPM" : "-- BPM";

    if (latestData.mode === "BREATHING") {
      modeTag.innerText = "MODE: CONTACTLESS RESPIRATION & VITALS SCANNER // COM3";
      document.getElementById('rightTitle').innerText = "Live Chest Inhale/Exhale Sinusoid (0.15 - 0.45 Hz) & Subcarriers";
    } else {
      modeTag.innerText = "MODE: THROUGH-WALL MOTION RADAR // COM3";
      document.getElementById('rightTitle').innerText = "Live Motion Confidence Timeline & 50-Subcarrier Spectrum";
    }

    if (latestData.calibrating) {
      banner.className = 'status-banner cal';
      stText.innerText = latestData.status;
      stText.style.color = '#f59e0b';
      dot.style.background = '#f59e0b';
    } else if (latestData.mode === "BREATHING") {
      if (latestData.human_detected) {
        banner.className = 'status-banner alert';
        stText.innerText = latestData.status;
        stText.style.color = '#ef4444';
        dot.style.background = '#ef4444';
      } else {
        banner.className = 'status-banner vitals';
        stText.innerText = latestData.status;
        stText.style.color = '#22d3ee';
        dot.style.background = '#22d3ee';
      }
    } else if (latestData.human_detected) {
      banner.className = 'status-banner alert';
      stText.innerText = '🚨 HUMAN PRESENCE DETECTED THROUGH WALL! 🚨';
      stText.style.color = '#ef4444';
      dot.style.background = '#ef4444';
    } else {
      banner.className = 'status-banner clear';
      stText.innerText = latestData.status;
      stText.style.color = '#10b981';
      dot.style.background = '#10b981';
    }
  } catch (e) {}
}
setInterval(pollState, 60);

function renderLoop() {
  const w = radarCanvas.width, h = radarCanvas.height;
  const cx = w / 2, cy = h / 2, maxR = Math.min(cx, cy) - 20;
  rCtx.fillStyle = 'rgba(6, 9, 15, 0.25)';
  rCtx.fillRect(0, 0, w, h);

  rCtx.strokeStyle = '#1f2937';
  rCtx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    rCtx.beginPath();
    rCtx.arc(cx, cy, (maxR / 4) * i, 0, Math.PI * 2);
    rCtx.stroke();
  }
  rCtx.beginPath();
  rCtx.moveTo(cx - maxR, cy); rCtx.lineTo(cx + maxR, cy);
  rCtx.moveTo(cx, cy - maxR); rCtx.lineTo(cx, cy + maxR);
  rCtx.stroke();

  sweepAngle += 0.04;
  const isVitals = latestData && latestData.mode === "BREATHING";
  const isAlert = latestData && latestData.human_detected;
  const sweepColor = isAlert ? 'rgba(239, 68, 68, 0.55)' : (isVitals ? 'rgba(6, 182, 212, 0.5)' : 'rgba(16, 185, 129, 0.45)');
  
  rCtx.beginPath();
  rCtx.moveTo(cx, cy);
  rCtx.arc(cx, cy, maxR, sweepAngle, sweepAngle + 0.35);
  rCtx.closePath();
  rCtx.fillStyle = sweepColor;
  rCtx.fill();

  if (latestData) {
    if (isVitals && latestData.breathing_bpm > 0 && !isAlert) {
      // Draw expanding/contracting lungs ring in Vitals Mode
      const waveArr = latestData.breath_wave;
      const latestWave = waveArr.length > 0 ? waveArr[waveArr.length - 1] : 0;
      const lungR = maxR * 0.45 + latestWave * 28;
      rCtx.beginPath();
      rCtx.arc(cx, cy, Math.max(15, Math.min(maxR, lungR)), 0, Math.PI * 2);
      rCtx.strokeStyle = '#22d3ee';
      rCtx.lineWidth = 4;
      rCtx.stroke();
      rCtx.fillStyle = 'rgba(6, 182, 212, 0.18)';
      rCtx.fill();
    } else if (isAlert) {
      const ratio = Math.min(1.0, latestData.confidence_score / (latestData.threshold * 2.2));
      const pulseR = 14 + Math.sin(Date.now() * 0.012) * 8 + ratio * 25;
      rCtx.beginPath();
      rCtx.arc(cx, cy - maxR * 0.35, pulseR, 0, Math.PI * 2);
      rCtx.fillStyle = 'rgba(239, 68, 68, 0.35)';
      rCtx.fill();
      rCtx.beginPath();
      rCtx.arc(cx, cy - maxR * 0.35, 9, 0, Math.PI * 2);
      rCtx.fillStyle = '#ef4444';
      rCtx.fill();
    }
  }

  // Right Panel Graph
  const gw = graphCanvas.width, gh = graphCanvas.height;
  gCtx.clearRect(0, 0, gw, gh);
  const splitY = gh * 0.62;

  if (latestData) {
    if (latestData.mode === "BREATHING") {
      // Draw Zero-Centered Chest Breathing Sinusoid
      const wave = latestData.breath_wave;
      const midY = splitY / 2;
      gCtx.strokeStyle = '#1f2937';
      gCtx.beginPath();
      gCtx.moveTo(0, midY); gCtx.lineTo(gw, midY);
      gCtx.stroke();

      gCtx.beginPath();
      for (let i = 0; i < wave.length; i++) {
        const x = (i / (wave.length - 1)) * gw;
        const y = midY - wave[i] * (midY - 18);
        if (i === 0) gCtx.moveTo(x, y);
        else gCtx.lineTo(x, y);
      }
      gCtx.strokeStyle = '#10b981';
      gCtx.lineWidth = 3.5;
      gCtx.stroke();
    } else {
      // Draw Standard Motion Z-Score Timeline
      const hist = latestData.history;
      const maxY = Math.max(10.0, latestData.threshold * 2.2);
      const thY = splitY - (latestData.threshold / maxY) * (splitY - 20);
      gCtx.strokeStyle = '#f59e0b';
      gCtx.setLineDash([6, 6]);
      gCtx.beginPath();
      gCtx.moveTo(0, thY); gCtx.lineTo(gw, thY);
      gCtx.stroke();
      gCtx.setLineDash([]);

      gCtx.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const x = (i / (hist.length - 1)) * gw;
        const val = Math.min(maxY, hist[i]);
        const y = splitY - (val / maxY) * (splitY - 20);
        if (i === 0) gCtx.moveTo(x, y);
        else gCtx.lineTo(x, y);
      }
      gCtx.strokeStyle = latestData.human_detected ? '#ef4444' : '#06b6d4';
      gCtx.lineWidth = 3;
      gCtx.stroke();
    }

    // Bottom 50 Subcarrier Bars
    const subs = latestData.subcarriers;
    const barW = (gw / subs.length) - 2;
    for (let i = 0; i < subs.length; i++) {
      const barH = Math.min(gh - splitY - 15, subs[i] * 18);
      const bx = i * (barW + 2);
      const by = gh - barH;
      gCtx.fillStyle = subs[i] > 2.0 ? (latestData.mode === "BREATHING" ? '#22d3ee' : '#ef4444') : '#3b82f6';
      gCtx.fillRect(bx, by, barW, barH);
    }
  }
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);
</script>
</body>
</html>
"""

class RadarHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/state":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(radar_state).encode("utf-8"))
        elif self.path.startswith("/set_threshold"):
            try:
                val = float(self.path.split("val=")[1])
                radar_state["threshold"] = val
            except Exception:
                pass
            self.send_response(200)
            self.end_headers()
        elif self.path == "/recalibrate":
            radar_state["trigger_recal"] = True
            self.send_response(200)
            self.end_headers()
        elif self.path == "/start_vitals":
            radar_state["trigger_vitals"] = True
            self.send_response(200)
            self.end_headers()

def parse_csi_amplitudes(line: str):
    if "CSI_DATA" not in line:
        return None
    match = re.search(r'\[([-\d,\s]+)\]', line)
    if not match:
        return None
    raw_str = match.group(1).replace(',', ' ')
    raw_iq = np.fromstring(raw_str, dtype=int, sep=' ')
    if len(raw_iq) < 64:
        return None
    if len(raw_iq) % 2 != 0:
        raw_iq = raw_iq[:-1]

    iq_pairs = raw_iq.reshape(-1, 2)
    amplitudes = np.sqrt(iq_pairs[:, 0]**2 + iq_pairs[:, 1]**2)

    if len(amplitudes) >= 64:
        valid_idx = list(range(6, 31)) + list(range(34, 59))
        amplitudes = amplitudes[valid_idx]
    elif len(amplitudes) > 20:
        amplitudes = amplitudes[6:-5]

    norm = np.linalg.norm(amplitudes)
    if norm < 1e-6:
        return None
    return (amplitudes / norm) * 100.0

def extract_respiration_numpy(breath_mat, fs=100.0):
    """
    Pure-NumPy Respiration Extractor:
    1. Selects top 8 chest-sensitive subcarriers & aligns polarity.
    2. Detrends slow drift (3s moving average) and smooths high-freq noise (0.35s moving average).
    3. Computes Zero-Padded 4096-point Hann FFT for 0.9 BPM frequency resolution.
    """
    N, num_subs = breath_mat.shape
    if N < 250:
        return 0.0, 0, [0.0] * 150

    # Zero-mean each subcarrier
    centered = breath_mat - np.mean(breath_mat, axis=0)

    # Pick top 8 subcarriers with strongest periodic energy
    variances = np.var(centered[-300:, :], axis=0)
    top_idx = np.argsort(variances)[-8:]
    selected = centered[:, top_idx]

    # Align phase polarity of the 8 subcarriers to the strongest one so they don't cancel out
    ref_sub = selected[:, -1]
    combined = np.zeros(N)
    for k in range(selected.shape[1]):
        sign = 1.0 if np.dot(selected[:, k], ref_sub) >= 0 else -1.0
        combined += sign * selected[:, k]
    combined /= selected.shape[1]

    # Bandpass via Pure NumPy Moving Averages:
    # Remove slow baseline drift (>3.0s window)
    k_drift = min(300, N // 2)
    drift = np.convolve(combined, np.ones(k_drift)/k_drift, mode='same')
    detrended = combined - drift

    # Smooth out high-frequency RF static (0.35s window -> ~2.8 Hz lowpass)
    k_smooth = 35
    breath_wave = np.convolve(detrended, np.ones(k_smooth)/k_smooth, mode='same')

    # Normalize waveform to [-1.0, 1.0] for live UI display (last 3 seconds = 300 packets, decimated to 150)
    recent_wave = breath_wave[-300::2]
    max_amp = np.percentile(np.abs(recent_wave), 95) + 1e-5
    ui_wave = np.clip(recent_wave / max_amp, -1.0, 1.0).tolist()

    # High-Resolution Zero-Padded FFT on the last 6 to 8 seconds
    win_sig = breath_wave * np.hanning(N)
    fft_mag = np.abs(np.fft.rfft(win_sig, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0/fs)

    # Human respiration band: 0.16 Hz to 0.45 Hz (9.6 to 27.0 Breaths Per Minute)
    mask = (freqs >= 0.16) & (freqs <= 0.45)
    if not np.any(mask):
        return 0.0, 0, ui_wave

    band_mag = fft_mag[mask]
    band_freqs = freqs[mask]
    peak_idx = np.argmax(band_mag)
    peak_f = band_freqs[peak_idx]
    
    # Calculate Spectral SNR (Peak power vs mean floor in band)
    snr = float(band_mag[peak_idx] / (np.mean(band_mag) + 1e-6))
    conf_pct = min(100, max(0, int((snr - 1.3) * 45)))
    bpm = round(float(peak_f * 60.0), 1)

    return bpm, conf_pct, [round(x, 3) for x in ui_wave]

def connect_and_find_baud(port):
    for baud in [921600, 115200]:
        print(f"Checking {port} at {baud} baud...")
        try:
            ser = serial.Serial(port, baud, timeout=0.2)
            ser.dtr = False
            ser.rts = False
            time.sleep(0.2)
            ser.reset_input_buffer()
            start = time.time()
            while time.time() - start < 2.5:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "CSI_DATA" in line:
                    print(f"✅ Locked onto CSI_DATA stream on {port} @ {baud} baud!")
                    return ser
            ser.close()
        except Exception as e:
            print(f"❌ Could not open {port}: {e}")
            return None
    return None

def csi_radar_worker():
    ser = connect_and_find_baud(SERIAL_PORT)
    if ser is None:
        radar_state["status"] = f"ERROR: NO CSI_DATA ON {SERIAL_PORT}"
        return

    radar_state["connected"] = True
    buffer = deque(maxlen=WINDOW_SIZE)
    breath_buffer = deque(maxlen=BREATH_WINDOW)
    cal_matrices = []
    
    cal_start = time.time()
    sub_base_mean = None
    sub_base_std = None
    smoothed_z = 0.0
    smoothed_bpm = 0.0
    history_z = deque([0.0]*120, maxlen=120)
    hit_streak = 0
    miss_streak = 0

    while True:
        try:
            # Button 1: Standard Empty-Room Motion Calibration
            if radar_state["trigger_recal"]:
                radar_state["trigger_recal"] = False
                radar_state["mode"] = "MOTION"
                radar_state["calibrating"] = True
                radar_state["human_detected"] = False
                cal_matrices.clear()
                cal_start = time.time()

            # Button 2: Vitals / Breathing Calibration (Locks onto stationary person)
            if radar_state["trigger_vitals"]:
                radar_state["trigger_vitals"] = False
                radar_state["mode"] = "BREATHING"
                radar_state["calibrating"] = True
                radar_state["human_detected"] = False
                breath_buffer.clear()
                cal_matrices.clear()
                cal_start = time.time()

            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            amps = parse_csi_amplitudes(raw_line)
            if amps is None:
                continue

            # Hampel Outlier Rejection
            if len(buffer) >= 5:
                recent_med = np.median(np.array(buffer)[-5:], axis=0)
                if recent_med.shape == amps.shape:
                    spike_mask = np.abs(amps - recent_med) > 2.5
                    amps = np.where(spike_mask, recent_med, amps)

            buffer.append(amps)
            breath_buffer.append(amps)
            if len(buffer) < WINDOW_SIZE:
                continue

            mat = np.array(buffer)
            slow_std = np.std(mat, axis=0)
            fast_diff = np.std(mat[-15:, :] - np.mean(mat[-30:-15, :], axis=0), axis=0)
            subcarrier_activity = 0.6 * slow_std + 0.4 * fast_diff

            # --- CALIBRATION PHASE (Handles both Motion & Vitals Lock) ---
            if radar_state["calibrating"]:
                cal_matrices.append(subcarrier_activity)
                elapsed = time.time() - cal_start
                target_dur = VITALS_LOCK_SECONDS if radar_state["mode"] == "BREATHING" else CALIBRATION_SECONDS
                rem = max(0, int(target_dur - elapsed))
                radar_state["cal_remaining"] = rem

                if radar_state["mode"] == "BREATHING":
                    radar_state["status"] = f"🫁 LOCKING ONTO SUBJECT CHEST... ({rem}s) — STAY STILL & BREATHE NORMALLY"
                else:
                    radar_state["status"] = f"⏳ CALIBRATING EMPTY ROOM... ({rem}s) — STEP BACK"

                if elapsed >= target_dur and len(cal_matrices) > 10:
                    cal_arr = np.array(cal_matrices)
                    sub_base_mean = np.percentile(cal_arr, 60, axis=0)
                    sub_base_std = np.maximum(np.std(cal_arr, axis=0), sub_base_mean * 0.18)
                    smoothed_z = 0.0
                    smoothed_bpm = 15.0
                    radar_state["calibrating"] = False
                continue

            # Compute Z-Score Activity
            sub_z_scores = np.maximum(0.0, (subcarrier_activity - sub_base_mean) / (sub_base_std + 1e-5))
            top_k = max(5, int(len(sub_z_scores) * 0.35))
            raw_z = float(np.mean(np.sort(sub_z_scores)[-top_k:]))
            smoothed_z = (0.28 * raw_z) + (0.72 * smoothed_z)
            history_z.append(round(smoothed_z, 2))

            # --- MODE A: DEDICATED VITALS / BREATHING SCANNER ---
            if radar_state["mode"] == "BREATHING":
                bpm, conf_pct, ui_wave = extract_respiration_numpy(np.array(breath_buffer))
                radar_state["breath_wave"] = ui_wave
                radar_state["breathing_conf"] = conf_pct

                if smoothed_z > radar_state["threshold"] * 1.3:
                    # Subject is walking/moving too much to isolate millimeter chest motion
                    radar_state["human_detected"] = True
                    radar_state["status"] = "⚠️ SUBJECT MOVING — HOLD STILL FOR RESPIRATION LOCK"
                else:
                    radar_state["human_detected"] = False
                    if bpm > 0 and conf_pct >= 25:
                        smoothed_bpm = (0.2 * bpm) + (0.8 * smoothed_bpm)
                        radar_state["breathing_bpm"] = round(smoothed_bpm, 1)
                        radar_state["status"] = f"🫁 RESPIRATION LOCKED: {radar_state['breathing_bpm']} BREATHS/MIN ({conf_pct}% SNR)"
                    else:
                        radar_state["status"] = f"🫁 GATHERING BREATH CYCLES... ({len(breath_buffer)}/{BREATH_WINDOW} PKTS)"

            # --- MODE B: STANDARD THROUGH-WALL MOTION RADAR ---
            else:
                thresh = radar_state["threshold"]
                if not radar_state["human_detected"]:
                    if smoothed_z > thresh:
                        hit_streak += 1
                        if hit_streak >= 4:
                            radar_state["human_detected"] = True
                            miss_streak = 0
                    else:
                        hit_streak = max(0, hit_streak - 1)
                else:
                    if smoothed_z < (thresh * 0.65):
                        miss_streak += 1
                        if miss_streak >= 10:
                            radar_state["human_detected"] = False
                            hit_streak = 0
                    else:
                        miss_streak = 0

                radar_state["status"] = "✅ SECTOR CLEAR (EMPTY ROOM)"

            radar_state["confidence_score"] = round(smoothed_z, 2)
            radar_state["subcarriers"] = [round(float(x), 2) for x in sub_z_scores]
            radar_state["history"] = list(history_z)

        except Exception:
            pass

if __name__ == "__main__":
    t = threading.Thread(target=csi_radar_worker, daemon=True)
    t.start()

    server = HTTPServer(("127.0.0.1", WEB_PORT), RadarHTTPHandler)
    url = f"http://localhost:{WEB_PORT}"
    print("\n" + "="*60)
    print(f" 🚀 TACTICAL & VITALS RADAR LIVE AT: {url}")
    print("="*60)
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down radar server...")
        server.server_close()