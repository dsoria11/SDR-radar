#!/usr/bin/env python3
"""
FMCW radar target detector for the bladeRF 2.0 micro (single file).

Each measurement cycle:
  1. TX a frame of NUM_CHIRPS back-to-back LFM chirps.
  2. Capture an RX block long enough to be guaranteed to contain that frame.
  3. Find the frame in the RX block from the TX->RX leakage
     (cross-correlation peak against the transmitted frame).
  4. Dechirp, coherently integrate the chirps, range FFT, noise floor, CA-CFAR.
  5. Hand the results to the GUI, which only redraws the visible tab.

Hookup: TX1 and RX1 to two antennas (or TX -> attenuator -> RX for a bench test).
Range 0 m = the leakage path (subtract antenna spacing / cable with RANGE_OFFSET_M).

Requires: numpy, pyqtgraph (+ PyQt5/PyQt6/PySide6), bladeRF Python bindings.
"""
import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from bladerf import _bladerf

# ============================ TUNABLE PARAMETERS ============================
# RF front end
CENTER_FREQ    = 5.8e9      # Hz
SAMPLE_RATE    = 30e6       # complex samples/s (<= 61.44e6)
RF_BANDWIDTH   = 30e6       # analog filter bandwidth, Hz
TX_GAIN        = 40         # dB
RX_GAIN        = 30         # dB (manual gain)

# Waveform
SWEEP_BW       = 28e6       # Hz, keep < SAMPLE_RATE. Range resolution = c / (2 * SWEEP_BW)
CHIRP_DURATION = 100e-6     # s
NUM_CHIRPS     = 64         # chirps per frame (coherently integrated)
TX_AMPLITUDE   = 0.7        # fraction of DAC full scale

# Processing / display
FFT_PAD        = 8          # range-FFT zero-padding factor (1 resolution cell = FFT_PAD bins)
RANGE_WINDOW   = "blackman" # "hann" (-31 dB sidelobes), "blackman" (-58 dB), "blackmanharris" (-92 dB)
REMOVE_DC      = True       # subtract beat-signal mean: kills zero-range leakage + RX DC offset
MAX_RANGE_M    = 50.0       # displayed range
RANGE_OFFSET_M = 0.0        # fixed offset subtracted from all ranges
MIN_RANGE_M    = 2.0        # ignore detections closer than this (leakage region)
WATERFALL_LEN  = 200        # frames of waterfall history
SYNC_MIN_CORR  = 0.1        # normalized leakage-correlation below this = sync miss

# CFAR (units: FFT bins; one range-resolution cell = FFT_PAD bins)
CFAR_GUARD     = 16         # guard cells on each side (cover the window's main lobe)
CFAR_TRAIN     = 24         # training cells on each side
CFAR_PFA       = 1e-4       # probability of false alarm
CFAR_MODE      = "SO"       # "CA" cell-averaging, "SO" smallest-of (robust next to the
                            # TX leakage peak / clutter edges), "GO" greatest-of

# Streaming
RX_BUFFER_SIZE = 16384      # samples per USB buffer (multiple of 1024)
RX_NUM_BUFFERS = 32
RX_TRANSFERS   = 16
TX_BUFFER_SIZE = 8192       # small TX queue keeps TX latency low
TX_NUM_BUFFERS = 8
TX_TRANSFERS   = 4
RX_BLOCK       = 65536      # samples per sync_rx call
SYNC_MARGIN    = 400_000    # extra RX samples captured to cover TX+RX stream latency
GUI_PERIOD_MS  = 30
# ============================================================================

C = 299_792_458.0

# ---- derived constants ----
NS = int(round(CHIRP_DURATION * SAMPLE_RATE))           # samples per chirp
FRAME_LEN = NS * NUM_CHIRPS
SLOPE = SWEEP_BW / (NS / SAMPLE_RATE)                   # Hz/s
CAPTURE_LEN = int(np.ceil((FRAME_LEN + SYNC_MARGIN) / RX_BLOCK)) * RX_BLOCK
NFFT = NS * FFT_PAD
RANGE_PER_BIN = (SAMPLE_RATE / NFFT) * C / (2 * SLOPE)
N_BINS = min(int((MAX_RANGE_M + RANGE_OFFSET_M) / RANGE_PER_BIN) + 1, NFFT // 2)
RANGES = np.arange(N_BINS) * RANGE_PER_BIN - RANGE_OFFSET_M
T_RAW = np.arange(CAPTURE_LEN) / SAMPLE_RATE

t = np.arange(NS) / SAMPLE_RATE
CHIRP = np.exp(1j * 2 * np.pi * (-SWEEP_BW / 2 * t + 0.5 * SLOPE * t ** 2)).astype(np.complex64)
FRAME = np.tile(CHIRP, NUM_CHIRPS)

_w = 2 * np.pi * np.arange(NS) / (NS - 1)
WINDOW = {"hann": np.hanning(NS),
          "blackman": np.blackman(NS),
          "blackmanharris": 0.35875 - 0.48829 * np.cos(_w) + 0.14128 * np.cos(2 * _w)
                            - 0.01168 * np.cos(3 * _w)}[RANGE_WINDOW].astype(np.float32)
WINDOW_GAIN = WINDOW.sum()

NFFT_SYNC = 1 << (CAPTURE_LEN - 1).bit_length()
TEMPLATE_FFT_CONJ = np.conj(np.fft.fft(FRAME, NFFT_SYNC))
TEMPLATE_NORM = np.linalg.norm(FRAME)

CFAR_N = 2 * CFAR_TRAIN if CFAR_MODE == "CA" else CFAR_TRAIN
CFAR_ALPHA = CFAR_N * (CFAR_PFA ** (-1.0 / CFAR_N) - 1)   # CA formula (approximate for SO/GO)
_i = np.arange(NFFT // 2)
_n = NFFT // 2
CFAR_L = (np.clip(_i - CFAR_GUARD - CFAR_TRAIN, 0, _n), np.clip(_i - CFAR_GUARD, 0, _n))
CFAR_R = (np.clip(_i + CFAR_GUARD + 1, 0, _n), np.clip(_i + CFAR_GUARD + CFAR_TRAIN + 1, 0, _n))


def cfar_threshold(power):
    cs = np.concatenate(([0.0], np.cumsum(power)))
    lc, rc = CFAR_L[1] - CFAR_L[0], CFAR_R[1] - CFAR_R[0]
    ls, rs = cs[CFAR_L[1]] - cs[CFAR_L[0]], cs[CFAR_R[1]] - cs[CFAR_R[0]]
    if CFAR_MODE == "CA":
        noise = (ls + rs) / (lc + rc)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            lm = np.where(lc > 0, ls / lc, np.nan)
            rm = np.where(rc > 0, rs / rc, np.nan)
        noise = np.fmin(lm, rm) if CFAR_MODE == "SO" else np.fmax(lm, rm)
    return CFAR_ALPHA * noise


# ============================== HARDWARE ====================================
class Radio:
    """Continuous TX (zeros or a radar frame on request) and continuous RX (captured on request)."""

    def __init__(self):
        self.sdr = _bladerf.BladeRF()
        self.rx = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.tx = self.sdr.Channel(_bladerf.CHANNEL_TX(0))
        for ch in (self.rx, self.tx):
            ch.frequency = int(CENTER_FREQ)
            ch.sample_rate = int(SAMPLE_RATE)
            ch.bandwidth = int(RF_BANDWIDTH)
        self.rx.gain_mode = _bladerf.GainMode.Manual
        self.rx.gain = RX_GAIN
        self.tx.gain = TX_GAIN

        self.sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1, fmt=_bladerf.Format.SC16_Q11,
                             num_buffers=RX_NUM_BUFFERS, buffer_size=RX_BUFFER_SIZE,
                             num_transfers=RX_TRANSFERS, stream_timeout=3500)
        self.sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, fmt=_bladerf.Format.SC16_Q11,
                             num_buffers=TX_NUM_BUFFERS, buffer_size=TX_BUFFER_SIZE,
                             num_transfers=TX_TRANSFERS, stream_timeout=3500)

        # Radar frame as interleaved SC16_Q11, padded to whole TX buffers
        self.tx_len = int(np.ceil(FRAME_LEN / TX_BUFFER_SIZE)) * TX_BUFFER_SIZE
        iq = np.zeros(2 * self.tx_len, np.int16)
        iq[0:2 * FRAME_LEN:2] = np.round(FRAME.real * TX_AMPLITUDE * 2047)
        iq[1:2 * FRAME_LEN:2] = np.round(FRAME.imag * TX_AMPLITUDE * 2047)
        self.tx_frame = bytearray(iq.tobytes())
        self.tx_zeros = bytearray(4 * TX_BUFFER_SIZE)

        self.running = True
        self.tx_request = False
        self.capturing = False
        self.chunks = []
        self.done = threading.Event()

        self.rx.enable = True
        self.tx.enable = True
        self.threads = [threading.Thread(target=self._rx_loop, daemon=True),
                        threading.Thread(target=self._tx_loop, daemon=True)]
        for th in self.threads:
            th.start()

    def _tx_loop(self):
        while self.running:
            if self.tx_request:
                self.tx_request = False
                self.sdr.sync_tx(self.tx_frame, self.tx_len)
            else:
                self.sdr.sync_tx(self.tx_zeros, TX_BUFFER_SIZE)

    def _rx_loop(self):
        buf = bytearray(4 * RX_BLOCK)
        while self.running:
            self.sdr.sync_rx(buf, RX_BLOCK)
            if self.capturing:
                self.chunks.append(np.frombuffer(buf, np.int16).copy())
                if len(self.chunks) * RX_BLOCK >= CAPTURE_LEN:
                    self.capturing = False
                    self.done.set()

    def capture(self):
        """One measurement cycle: start RX capture, fire one TX frame, return complex64 RX block."""
        self.chunks = []
        self.done.clear()
        self.capturing = True
        self.tx_request = True
        if not self.done.wait(2.0):
            return None
        raw = np.concatenate(self.chunks).astype(np.float32) * (1.0 / 2048.0)
        return raw.view(np.complex64)          # interleaved I/Q -> complex

    def close(self):
        self.running = False
        for th in self.threads:
            th.join(timeout=4)
        self.rx.enable = False
        self.tx.enable = False
        self.sdr.close()


# ================================ DSP =======================================
def process(iq):
    # 1) Leakage sync: correlate against the full TX frame -> unique peak at frame start
    corr = np.fft.ifft(np.fft.fft(iq, NFFT_SYNC) * TEMPLATE_FFT_CONJ)[:len(iq) - FRAME_LEN + 1]
    mag = np.abs(corr)
    start = int(np.argmax(mag))
    seg = iq[start:start + FRAME_LEN]
    quality = mag[start] / (np.linalg.norm(seg) * TEMPLATE_NORM + 1e-12)

    # 2) Dechirp (ref * conj(rx) -> beat at +slope*delay), coherent integration over chirps
    beat = (seg.reshape(NUM_CHIRPS, NS).conj() * CHIRP).mean(axis=0)
    if REMOVE_DC:
        beat = beat - beat.mean()

    # 3) Fast-time range FFT
    spec = np.fft.fft(beat * WINDOW, NFFT)[:NFFT // 2] / WINDOW_GAIN
    power = np.abs(spec) ** 2 + 1e-20

    # 4) Dynamic noise floor + CA-CFAR
    noise_db = 10 * np.log10(np.median(power))
    thresh = cfar_threshold(power)

    p, th = power[:N_BINS], thresh[:N_BINS]
    peak = np.zeros(N_BINS, bool)
    peak[1:-1] = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])
    det = np.flatnonzero(peak & (p > th) & (RANGES >= MIN_RANGE_M))

    adc_peak = float(np.max(np.abs(iq.view(np.float32))))
    return dict(iq=iq, start=start, quality=quality,
                prof_db=10 * np.log10(p), thr_db=10 * np.log10(th), noise_db=noise_db,
                det_idx=det, adc_peak=adc_peak)


class Radar(threading.Thread):
    """Measurement loop: capture -> sync -> DSP -> publish latest result for the GUI."""

    def __init__(self, radio):
        super().__init__(daemon=True)
        self.radio = radio
        self.running = True
        self.latest = None
        self.history = None
        self.misses = 0

    def run(self):
        frame_id, fps, t_last = 0, 0.0, time.time()
        while self.running:
            iq = self.radio.capture()
            if iq is None:
                self.misses += 1
                continue
            r = process(iq)
            r["synced"] = r["quality"] >= SYNC_MIN_CORR
            if r["synced"]:
                if self.history is None:
                    self.history = np.tile(r["prof_db"], (WATERFALL_LEN, 1)).astype(np.float32)
                self.history[:-1] = self.history[1:]
                self.history[-1] = r["prof_db"]
            else:
                self.misses += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 / (now - t_last)
            t_last = now
            frame_id += 1
            r.update(id=frame_id, fps=fps, misses=self.misses,
                     history=None if self.history is None else self.history.copy())
            self.latest = r


# ================================ GUI =======================================
class RadarWindow(QtWidgets.QMainWindow):
    def __init__(self, radar, radio):
        super().__init__()
        self.radar, self.radio = radar, radio
        self.shown_id = -1
        self.setWindowTitle("bladeRF FMCW Radar")
        self.resize(1200, 750)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        dash = pg.mkPen((200, 80, 80), style=QtCore.Qt.PenStyle.DashLine)

        # Tab 1: raw I/Q
        self.p_raw = pg.PlotWidget(title="Raw RX capture (ADC, full scale = ±1)")
        self.p_raw.setLabel("bottom", "Time", units="s")
        self.p_raw.setLabel("left", "Amplitude", units="FS")
        self.p_raw.addLegend()
        self.p_raw.setDownsampling(auto=True, mode="peak")
        self.p_raw.setClipToView(True)
        self.c_i = self.p_raw.plot(pen="c", name="I")
        self.c_q = self.p_raw.plot(pen="m", name="Q")
        for y in (-1, 1):
            self.p_raw.addLine(y=y, pen=dash)
        self.sync_lines = [pg.InfiniteLine(angle=90, pen="y") for _ in range(2)]
        for ln in self.sync_lines:
            self.p_raw.addItem(ln)
        self.tabs.addTab(self.p_raw, "Raw Signal")

        # Tab 2: range profile
        self.p_rng = pg.PlotWidget(title="1D Range Profile")
        self.p_rng.setLabel("bottom", "Range", units="m")
        self.p_rng.setLabel("left", "Magnitude", units="dBFS")
        self.p_rng.showGrid(x=True, y=True, alpha=0.3)
        self.c_rng = self.p_rng.plot(pen="y")
        self.nf_line = self.p_rng.addLine(y=-100, pen=dash)
        self.tabs.addTab(self.p_rng, "Range Profile")

        # Tab 3: waterfall
        self.p_wf = pg.PlotWidget(title="Range-Time Waterfall (newest at top)")
        self.p_wf.setLabel("bottom", "Range", units="m")
        self.p_wf.setLabel("left", "Frame")
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("viridis"))
        self.p_wf.addItem(self.img)
        self.wf_rect = QtCore.QRectF(RANGES[0], 0, N_BINS * RANGE_PER_BIN, WATERFALL_LEN)
        self.tabs.addTab(self.p_wf, "Waterfall")

        # Tab 4: CFAR overlay
        self.p_cfar = pg.PlotWidget(title=f"{CFAR_MODE}-CFAR (guard {CFAR_GUARD}, train {CFAR_TRAIN}, Pfa {CFAR_PFA:g})")
        self.p_cfar.setLabel("bottom", "Range", units="m")
        self.p_cfar.setLabel("left", "Magnitude", units="dBFS")
        self.p_cfar.showGrid(x=True, y=True, alpha=0.3)
        self.p_cfar.addLegend()
        self.c_prof = self.p_cfar.plot(pen="y", name="Range profile")
        self.c_thr = self.p_cfar.plot(pen=pg.mkPen("r", width=2), name="CFAR threshold")
        self.c_det = self.p_cfar.plot(pen=None, symbol="o", symbolBrush="g", symbolSize=10, name="Detections")
        self.tabs.addTab(self.p_cfar, "CFAR")

        self.status = QtWidgets.QLabel("Waiting for first frame...")
        self.statusBar().addWidget(self.status)

        self.tabs.currentChanged.connect(lambda _: self.refresh(force=True))
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(GUI_PERIOD_MS)

    def refresh(self, force=False):
        r = self.radar.latest
        if r is None or (r["id"] == self.shown_id and not force):
            return
        self.shown_id = r["id"]

        # Only touch the plot that is visible
        tab = self.tabs.currentIndex()
        if tab == 0:
            self.c_i.setData(T_RAW, r["iq"].real)
            self.c_q.setData(T_RAW, r["iq"].imag)
            self.sync_lines[0].setValue(r["start"] / SAMPLE_RATE)
            self.sync_lines[1].setValue((r["start"] + FRAME_LEN) / SAMPLE_RATE)
        elif tab == 1:
            self.c_rng.setData(RANGES, r["prof_db"])
            self.nf_line.setValue(r["noise_db"])
        elif tab == 2 and r["history"] is not None:
            h = r["history"]
            self.img.setImage(h, autoLevels=False, levels=(r["noise_db"] - 3, float(h.max())),
                              rect=self.wf_rect)
        elif tab == 3:
            self.c_prof.setData(RANGES, r["prof_db"])
            self.c_thr.setData(RANGES, r["thr_db"])
            d = r["det_idx"]
            self.c_det.setData(RANGES[d], r["prof_db"][d])

        # Status line (cheap, always)
        d = r["det_idx"]
        if len(d):
            best = d[np.argmax(r["prof_db"][d])]
            tgt = f"target {RANGES[best]:.2f} m (SNR {r['prof_db'][best] - r['noise_db']:.1f} dB)"
        else:
            tgt = "no target"
        clip = "  CLIPPING" if r["adc_peak"] >= 0.999 else ""
        sync = f"sync {r['quality']:.2f}" + ("" if r["synced"] else " LOST")
        self.status.setText(f"{r['fps']:.1f} fps | {sync} | noise {r['noise_db']:.1f} dBFS | "
                            f"ADC peak {r['adc_peak']:.2f}{clip} | {tgt} | misses {r['misses']}")

    def closeEvent(self, ev):
        self.timer.stop()
        self.radar.running = False
        self.radar.join(timeout=4)
        self.radio.close()
        ev.accept()


if __name__ == "__main__":
    print(f"Chirp: {NS} samples, frame: {FRAME_LEN} samples, capture: {CAPTURE_LEN} samples")
    print(f"Range resolution {C / (2 * SWEEP_BW):.2f} m, bin spacing {RANGE_PER_BIN:.3f} m, "
          f"max unambiguous range {SAMPLE_RATE / 2 * C / (2 * SLOPE):.0f} m")
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    app = pg.mkQApp("FMCW Radar")
    radio = Radio()
    radar = Radar(radio)
    radar.start()
    win = RadarWindow(radar, radio)
    win.show()
    app.exec()
