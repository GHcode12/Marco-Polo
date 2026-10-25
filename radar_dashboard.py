import sys
import re
import time
import serial
import numpy as np
from collections import deque
from scipy.signal import butter, lfilter, welch
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ================= CONFIGURATION =================
SERIAL_PORT = "COM3"  # <-- Make sure this matches your Receiver COM port!
FS = 100.0  # 100 Hz packet rate
HISTORY_SECONDS = 6  # 6 seconds rolling window (600 packets)
CALIBRATION_SECONDS = 6  # Baseline duration
MOTION_Z_TRIGGER = 3.5  # Z-Score threshold for motion
FALL_Z_TRIGGER = 9.5  # Z-Score spike threshold for sudden fall


# =================================================

class CSIRadarApp(QtWidgets.QMainWindow):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.win_size = int(FS * HISTORY_SECONDS)
        self.buffer = deque(maxlen=self.win_size)

        # Filters: Motion (0.7 - 5.0 Hz), Breathing (0.15 - 0.45 Hz)
        nyq = 0.5 * FS
        self.b_mot, self.a_mot = butter(3, [0.7 / nyq, 5.0 / nyq], btype='band')
        self.b_br, self.a_br = butter(2, [0.15 / nyq, 0.45 / nyq], btype='band')

        # Calibration & State
        self.calibrating = True
        self.cal_start = None
        self.baseline_history = []
        self.base_mean = 0.2
        self.base_std = 0.05
        self.smoothed_score = 0.0
        self.z_history = deque([0.0] * 200, maxlen=200)
        self.fall_cooldown = 0

        self._init_ui()

        # Timer to poll serial and update plots at ~30 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(30)

    def _init_ui(self):
        self.setWindowTitle("Wi-Fi CSI Through-Wall Human & Vital Radar")
        self.resize(1200, 780)
        self.setStyleSheet("background-color: #0d1117; color: #e6edf3;")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top Status Banner
        top_bar = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("📡 WARMING UP BUFFER...")
        self.status_label.setFont(QtGui.QFont("Segoe UI", 20, QtGui.QFont.Weight.Bold))
        self.status_label.setStyleSheet(
            "background-color: #161b22; color: #58a6ff; padding: 14px; border-radius: 8px; border: 2px solid #30363d;"
        )
        top_bar.addWidget(self.status_label, stretch=3)

        self.metrics_label = QtWidgets.QLabel("Z-Score: 0.0\nBreathing: -- BPM")
        self.metrics_label.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Weight.Bold))
        self.metrics_label.setStyleSheet(
            "background-color: #161b22; padding: 14px; border-radius: 8px; border: 1px solid #30363d;")
        top_bar.addWidget(self.metrics_label, stretch=1)

        self.cal_btn = QtWidgets.QPushButton("🔄 Recalibrate (Press C)")
        self.cal_btn.setFont(QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold))
        self.cal_btn.setStyleSheet("background-color: #238636; color: white; padding: 16px; border-radius: 8px;")
        self.cal_btn.clicked.connect(self.start_calibration)
        top_bar.addWidget(self.cal_btn)
        layout.addLayout(top_bar)

        # Graphics Grid
        pg.setConfigOptions(antialias=True, background='#0d1117', foreground='#8b949e')
        self.graph_win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_win)

        # 1. Waterfall Spectrogram
        self.p_heat = self.graph_win.addPlot(row=0, col=0, colspan=2, title="Live Wi-Fi OFDM Subcarrier Waterfall")
        self.p_heat.setLabel('left', 'Subcarrier Index')
        self.p_heat.setLabel('bottom', 'Time Samples')
        self.img_item = pg.ImageItem()
        self.p_heat.addItem(self.img_item)
        self.img_item.setColorMap(pg.colormap.get('inferno'))

        # 2. Motion Confidence Z-Score Timeline
        self.p_mot = self.graph_win.addPlot(row=1, col=0, title="Through-Wall Motion Confidence (Z-Score)")
        self.p_mot.setYRange(0, 12)
        self.mot_curve = self.p_mot.plot(pen=pg.mkPen('#ff7b72', width=2))
        self.thresh_line = pg.InfiniteLine(pos=MOTION_Z_TRIGGER, angle=0,
                                           pen=pg.mkPen('#ffa657', style=QtCore.Qt.PenStyle.DashLine, width=1.5))
        self.p_mot.addItem(self.thresh_line)

        # 3. Stationary Breathing Waveform
        self.p_breath = self.graph_win.addPlot(row=1, col=1,
                                               title="Extracted Chest Breathing Sinusoid (0.15 - 0.45 Hz)")
        self.breath_curve = self.p_breath.plot(pen=pg.mkPen('#3fb950', width=2))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_C:
            self.start_calibration()

    def start_calibration(self):
        self.calibrating = True
        self.cal_start = time.time()
        self.baseline_history.clear()
        self.fall_cooldown = 0

    def parse_csi_unit_vector(self, line: str):
        if "CSI_DATA" not in line:
            return None
        match = re.search(r'\[([-\d,\s]+)\]', line)
        if not match:
            return None
        try:
            raw_iq = np.fromstring(match.group(1).replace(',', ' '), dtype=int, sep=' ')
            if len(raw_iq) < 64:
                return None
            if len(raw_iq) % 2 != 0:
                raw_iq = raw_iq[:-1]

            iq = raw_iq.reshape(-1, 2)
            amps = np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)
            if len(amps) >= 64:
                valid_idx = list(range(6, 31)) + list(range(34, 59))
                amps = amps[valid_idx]
            elif len(amps) > 20:
                amps = amps[6:-5]

            norm = np.linalg.norm(amps)
            return (amps / norm) if norm > 1e-6 else None
        except Exception:
            return None

    def update_loop(self):
        try:
            # Drain serial queue safely
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                unit_amps = self.parse_csi_unit_vector(line)
                if unit_amps is not None:
                    if len(self.buffer) >= 5:
                        med = np.median(np.array(self.buffer)[-5:], axis=0)
                        if np.max(np.abs(unit_amps - med)) > 0.18:
                            unit_amps = med
                    self.buffer.append(unit_amps)
        except Exception as e:
            print(f"Serial read warning: {e}")

        if len(self.buffer) < self.win_size:
            self.status_label.setText(f"📡 FILLING RADAR BUFFER... ({len(self.buffer)}/{self.win_size})")
            return

        if self.cal_start is None:
            self.cal_start = time.time()

        try:
            mat = np.array(self.buffer)  # Shape: (600, N_subcarriers)
            mat_zero = mat - np.mean(mat, axis=0)

            # 1. Update Waterfall Heatmap safely using .T to match image dimensions
            heat_slice = np.abs(mat_zero[-150:, :]).T
            self.img_item.setImage(heat_slice, autoLevels=False, levels=(0.0, 0.035))

            # 2. Motion Filter (0.7 - 5.0 Hz)
            filt_mot = lfilter(self.b_mot, self.a_mot, mat_zero, axis=0)
            sub_energy = np.std(filt_mot[-50:, :], axis=0)
            sorted_e = np.sort(sub_energy)
            trimmed_e = sorted_e[int(len(sorted_e) * 0.15):int(len(sorted_e) * 0.85)]
            raw_score = float(np.mean(trimmed_e) * 1000.0)

            # Calibration Phase
            if self.calibrating:
                self.baseline_history.append(raw_score)
                rem = max(0.0, CALIBRATION_SECONDS - (time.time() - self.cal_start))
                self.status_label.setText(f"⏳ CALIBRATING EMPTY ROOM... {rem:.1f}s (STEP BACK!)")
                self.status_label.setStyleSheet(
                    "background-color: #1f2937; color: #facc15; padding: 14px; border-radius: 8px; border: 2px solid #eab308;")
                if rem <= 0 and len(self.baseline_history) > 10:
                    clean_b = np.sort(self.baseline_history)[:int(len(self.baseline_history) * 0.9)]
                    self.base_mean = float(np.mean(clean_b))
                    self.base_std = max(float(np.std(clean_b)), self.base_mean * 0.12, 0.04)
                    self.smoothed_score = self.base_mean
                    self.calibrating = False
                return

            # Motion Z-Score
            self.smoothed_score = (0.25 * raw_score) + (0.75 * self.smoothed_score)
            z_score = max(0.0, (self.smoothed_score - self.base_mean) / self.base_std)
            self.z_history.append(z_score)
            self.mot_curve.setData(list(self.z_history))

            # Breathing Extraction
            top_5_idx = np.argsort(np.var(mat_zero[-300:, :], axis=0))[-5:]
            sig_1d = np.mean(mat_zero[:, top_5_idx], axis=1)
            breath_wave = lfilter(self.b_br, self.a_br, sig_1d)
            self.breath_curve.setData(breath_wave[-400:])

            freqs, psd = welch(breath_wave[-500:], fs=FS, nperseg=min(400, len(breath_wave)))
            mask = (freqs >= 0.16) & (freqs <= 0.45)
            bpm = 0.0
            breath_snr = 0.0
            if np.any(mask):
                peak_f = freqs[mask][np.argmax(psd[mask])]
                breath_snr = np.max(psd[mask]) / (np.mean(psd[mask]) + 1e-9)
                bpm = round(float(peak_f * 60.0), 1)

            # State Machine Classification
            z_arr = np.array(self.z_history)
            recent_peak_z = np.max(z_arr[-45:-12]) if len(z_arr) > 50 else 0.0

            if self.fall_cooldown > 0:
                self.fall_cooldown -= 1
                state_text = "🚨 CRITICAL: SUDDEN FALL / IMPACT DETECTED! 🚨"
                style = "background-color: #7f1d1d; color: #fecaca; padding: 14px; border-radius: 8px; border: 2px solid #ef4444;"
            elif recent_peak_z > FALL_Z_TRIGGER and z_score < (MOTION_Z_TRIGGER * 0.85):
                self.fall_cooldown = 45
                state_text = "🚨 CRITICAL: SUDDEN FALL / IMPACT DETECTED! 🚨"
                style = "background-color: #7f1d1d; color: #fecaca; padding: 14px; border-radius: 8px; border: 2px solid #ef4444;"
            elif z_score > MOTION_Z_TRIGGER:
                state_text = "🏃 HUMAN MOTION DETECTED THROUGH WALL"
                style = "background-color: #451a03; color: #fdba74; padding: 14px; border-radius: 8px; border: 2px solid #f97316;"
                bpm = 0.0
            elif z_score > 1.1 and breath_snr > 2.1:
                state_text = f"🫁 STATIONARY HUMAN — BREATHING LOCKED ({bpm} BPM)"
                style = "background-color: #064e3b; color: #6ee7b7; padding: 14px; border-radius: 8px; border: 2px solid #10b981;"
            else:
                state_text = "✅ SECTOR CLEAR (EMPTY ROOM)"
                style = "background-color: #161b22; color: #8b949e; padding: 14px; border-radius: 8px; border: 2px solid #30363d;"
                bpm = 0.0

            self.status_label.setText(state_text)
            self.status_label.setStyleSheet(style)
            bpm_str = f"{bpm} BPM" if bpm > 0 else "-- BPM"
            self.metrics_label.setText(f"Confidence Z: {z_score:5.1f}\nBreathing:  {bpm_str}")

        except Exception as e:
            # Catch loop calculation errors safely without crashing the UI window
            print(f"Render loop error: {e}")


def connect_serial(port):
    for baud in [921600, 115200]:
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
            ser.dtr = False
            ser.rts = False
            time.sleep(0.2)
            ser.reset_input_buffer()
            t0 = time.time()
            while time.time() - t0 < 2.0:
                if "CSI_DATA" in ser.readline().decode('utf-8', errors='ignore'):
                    print(f"✅ Connected to {port} @ {baud} baud")
                    return ser
            ser.close()
        except Exception as e:
            print(f"Error opening {port}: {e}")
            return None
    return None


if __name__ == "__main__":
    ser = connect_serial(SERIAL_PORT)
    if ser is None:
        print(f"Could not find CSI_DATA on {SERIAL_PORT}. Check COM port & power on Sender board!")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    win = CSIRadarApp(ser)
    win.show()
    sys.exit(app.exec())