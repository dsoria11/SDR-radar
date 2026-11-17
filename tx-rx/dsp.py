"""
DSP.py
~~~~~~~

Contains the funtions to configure and start the bladeRF, process the radar echo, and to conduct the DSP thread from main.
"""

import numpy as np
import pandas as pd
import time
import json
import queue
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
from chirp_generator import gen_radar_chirp, ca_cfar_detect, ca_cfar_detect_gated
from bladerf import _bladerf
from trigger_ext import trigger_init, trigger_arm, trigger_fire, TriggerRole, TriggerSignal

def check_config(cfg):
    """Fail fast if the capture layout or buffer sizes cannot work."""
    fs = cfg["sample_rate"]
    n_chirp = int(round(fs * cfg["pulse_duration"]))
    n_rx = cfg["capture_samples"]
    n_off = cfg["tx_offset_samples"]
    buf = cfg["buffer_size"]

    assert cfg["rf_bandwidth"] >= cfg["chirp_bandwidth"], "RF filter narrower than chirp"
    assert cfg["chirp_bandwidth"] < fs, "chirp wider than sample rate"
    assert buf % 1024 == 0, "bladeRF buffer_size must be a multiple of 1024"
    assert n_rx % buf == 0, "capture/TX frame must be whole buffers or TX will stall"
    assert cfg["num_transfers"] < cfg["num_buffers"], "num_transfers must be < num_buffers"
    assert n_off + n_chirp + 1024 <= n_rx, "chirp + margin does not fit in the capture"

def configure_sdr(sdr, cfg):
    """ 
    Configures the bladeRF device with the specified settings from the configuration dictionary.
    Channels remain disabled here so main can enable/disable them per measurement.
    """
    tx_ch = sdr.Channel(_bladerf.CHANNEL_TX(0))
    rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))
    for ch in [tx_ch, rx_ch]:
        ch.frequency = int(cfg["center_frequency"])
        ch.sample_rate = int(cfg["sample_rate"])
        ch.bandwidth = int(cfg["rf_bandwidth"])

    rx_ch.gain_mode = _bladerf.GainMode.Manual
    rx_ch.gain = int(cfg["rx_gain"])
    tx_ch.gain = int(cfg["tx_gain"])

    sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=int(cfg["num_buffers"]), buffer_size=int(cfg["buffer_size"]), num_transfers=int(cfg["num_transfers"]), stream_timeout=int(cfg["stream_timeout"]))
    sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1, fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=int(cfg["num_buffers"]), buffer_size=int(cfg["buffer_size"]), num_transfers=int(cfg["num_transfers"]), stream_timeout=int(cfg["stream_timeout"]))

    # trigger setup
    tx_trig = trigger_init(sdr, _bladerf.CHANNEL_TX(0), TriggerSignal.MINI_EXP_1)
    rx_trig = trigger_init(sdr, _bladerf.CHANNEL_RX(0), TriggerSignal.MINI_EXP_1)

    return tx_ch, rx_ch, tx_trig, rx_trig

def unpack_sc16_q11(rx_bytes):
    """Converts raw SC16_Q11 bytes into normalized complex64 samples and measures peak/headroom."""
    rx_int16 = np.frombuffer(rx_bytes, dtype=np.int16)
    rx = (rx_int16[0::2].astype(np.float32) + 1j * rx_int16[1::2].astype(np.float32)) / 2048.0
    peak = float(np.max(np.abs(rx)))
    headroom_db = 20.0 * np.log10(1.0 / max(peak, 1e-12))
    return rx, peak, headroom_db


def find_leakage_index(rx, chirp):
    """Cross-correlates RX capture with the active chirp to verify trigger association."""
    corr = np.abs(signal.correlate(rx, chirp, mode="valid", method="fft"))
    time_zero_idx = int(np.argmax(corr))
    max_corr = float(corr[time_zero_idx])
    return time_zero_idx, max_corr


def dechirp_at(rx, chirp, start):
    """
    Multiply the reference chirp by the conjugate of the received samples,
    starting at a FIXED index (offset + measured latency), not the correlation peak.
    """
    seg = rx[start : start + chirp.size]
    return chirp * np.conj(seg)


def range_spectrum(mixed, fs, slope, pad_factor):
    """Hann window, zero-padded FFT, and the matching range axis in metres."""
    n_fft = pad_factor * mixed.size
    spec = np.abs(np.fft.fft(mixed * np.hanning(mixed.size), n_fft))[: n_fft // 2]
    ranges = np.arange(spec.size) * (fs / n_fft) * 3e8 / (2.0 * slope)
    return ranges, spec


def pick_peak(ranges, spec, max_range_m):
    """Strongest bin inside the range gate, refined with a 3-point parabola."""
    k_max = np.searchsorted(ranges, max_range_m)
    k = 1 + int(np.argmax(spec[1:k_max]))
    a, b, g = spec[k - 1], spec[k], spec[k + 1]
    denom = a - 2.0 * b + g
    shift = 0.5 * (a - g) / denom if denom != 0 else 0.0
    distance_m = float(ranges[k] + shift * (ranges[1] - ranges[0]))
    return distance_m, k, float(b)

def process_radar_echo(rx_bytes, tx_reference, sample_rate, pulse_duration, chirp_bandwidth, background_profile=None, max_range_m=10.0, use_background_subtraction=False):
    """ 
    Processes a single synchronized Rx capture buffer.
        1. Locates exact start sample of the chirp (time zero index)
        2. Mixesthe aligned Rx signal with the Tx reference
        3. Computes the FFT to find beat frequency and distance
    """
    rx_int16 = np.frombuffer(rx_bytes, dtype=np.int16)
    rx_complex = (rx_int16[0::2] + 1j * rx_int16[1::2]) / 2048.0
    rx_complex -= np.mean(rx_complex) 
    
    tx_active = tx_reference[np.abs(tx_reference) > 0.1]  

    # Locate chirp in the capture buffer
    correlation_alignment = np.abs(signal.correlate(rx_complex, tx_reference, mode='valid', method='fft'))
    max_align_corr = np.max(correlation_alignment)
    
    # Check if the correlation peak is strong enough to consider the chirp detected
    if max_align_corr < 120.0:
        print(f"[DSP] Chirp not found in capture")
        return np.zeros(10), 0.0, 0.0, rx_complex, 0.0, 0, 0, max_align_corr, background_profile

    time_zero_idx = int(np.argmax(correlation_alignment))
    rx_aligned = rx_complex[time_zero_idx : time_zero_idx + len(tx_active)]
    
    # FMCW Mixing
    mixed_signal = tx_active * np.conj(rx_aligned)
    mixed_signal -= np.mean(mixed_signal) # remove 0hz DC offset
    
    window = np.hanning(len(mixed_signal))
    mixed_windowed = mixed_signal * window
    
    # Fast Fourier Transform (FFT) and optional clutter subtraction
    pad_len = len(mixed_windowed) * 8 # 8x zero-padding for sub bin precision
    fft_result = np.fft.fft(mixed_windowed, n=pad_len)
    fft_mag = np.abs(fft_result)[:pad_len//2] 
    
    max_fft = np.max(fft_mag)
    if max_fft > 0:
        fft_mag = fft_mag / max_fft

    # --- Static clutter removal ---
    if use_background_subtraction:
        if background_profile is None:
            background_profile = fft_mag.copy()
        else:
            background_profile = 0.95 * background_profile + 0.05 * fft_mag
        fft_clean = np.maximum(fft_mag - background_profile, 0.0)
    else:
        fft_clean = fft_mag
        
    # Range-gated beat frequency and distance calculation
    bin_size_hz = sample_rate / pad_len
    start_bin = 1 

    # Convert max_range_m (10m) into an upper FFT bin limit so the DSP never searches beyond the maximum range
    max_beat_freq = (2 * chirp_bandwidth * max_range_m) / (3e8 * pulse_duration)
    max_bin = min(int(max_beat_freq / bin_size_hz), len(fft_clean))

    distance_m = 0.0
    beat_freq = 0.0
    target_score = 0.0
    target_idx = start_bin

    if start_bin < max_bin:
        search_window = fft_clean[start_bin:max_bin]

        # Compute noise floor from raw maginute inside physical range gate
        noise_floor = float(np.median(fft_mag[start_bin:max_bin]))
        if noise_floor <= 1e-6:
            noise_floor = 1e-6  # Avoid divide by zero error
        
        rel_idx = int(np.argmax(search_window))
        if 0 < rel_idx < len(search_window) - 1:
            alpha = search_window[rel_idx - 1]
            beta = search_window[rel_idx]
            gamma = search_window[rel_idx + 1]
            denom = alpha - 2 * beta + gamma
            shift = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            exact_idx = rel_idx + shift
        else:
            exact_idx = float(rel_idx)

        true_target_idx = start_bin + exact_idx
        beat_freq = true_target_idx * bin_size_hz
        distance_m = ((3e8 * beat_freq * pulse_duration) / (2 * chirp_bandwidth)) 

        peak_amplitude = float(search_window[rel_idx])
        target_score = 20.0 * np.log10((peak_amplitude + 1e-12) / noise_floor)
        target_idx = start_bin + rel_idx
     
    return fft_clean, distance_m, beat_freq, rx_aligned, target_score, target_idx, time_zero_idx, max_align_corr, background_profile

def run_dsp_process(stop_event, rx_queue, gui_queue, target_data_list, tx_reference, sample_rate, iq_queue):
    """  
    Standalone DSP process.
    """
    print("[DSP] Process is live.")

    frame_count = 0

    try: 
        with open("samples_config.json", "r") as file:
                data = json.load(file)
    except FileNotFoundError:
                print("Error: samples_config.json not found.")
                
    pulse_duration = data["pulse_duration"]
    chirp_bandwidth = data["chirp_bandwidth"]

    last_plot_time = 0 # initialize timer

    try:
        background_profile = None 

        while not stop_event.is_set():
            try:
                rx_bytes = rx_queue.get(timeout=1)  # Wait for data with a timeout

                #print("Debug: Received buffer...")

                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"[DSP] Frame count: {frame_count}.")

                correlation, distance_m, time_delay_ms, rx_aligned, target_score, target_idx, leakage_peak_idx, max_corr, background_profile = process_radar_echo(
                    rx_bytes, tx_reference, sample_rate, pulse_duration, 
                    chirp_bandwidth, background_profile
                )

                # Send live data to the Flask server
                if not iq_queue.full():
                    iq_queue.put(rx_aligned)

                #print(f"Debug: Math finished. Raw Peak: {max_corr:.2f}")

                if max_corr > 1.0:

                    confidence_threshold = 0.15 # Target must be atleast a 15% match
                    is_detected = (target_score > confidence_threshold) and (distance_m > 0)

                    if is_detected:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                        hit_data = {
                            "Timestamp": timestamp,
                            "Correlation_Score": round(target_score, 3),
                            "Distance_m": distance_m,
                            "Time_Delay_ms": time_delay_ms,
                            # "Threshold": threshold,
                            # "SNR": snr
                        }
                        print(f"[DSP] ({timestamp}) Target Detected | Range: {distance_m:.2f}m | SNR: {target_score:.2f} dB")

                        # Append to the shared multiprocessing.Manager list
                        target_data_list.append(hit_data)

                        if not gui_queue.full():
                            gui_queue.put(hit_data)

                        # --- Visual for correlation peaks and the waterfall plots ---
                        current_time = time.time()
                        if current_time - last_plot_time > 1.0:  
                            plt.figure(figsize=(10, 9))
                        
                        # 1. Spectrogram 
                        plt.subplot(2, 1, 1)
                        plt.title("1. Spectrogram (Waterfall / Frequency Sweep)")
                        plt.specgram(rx_aligned, NFFT=256, Fs=sample_rate, Fc=0, noverlap=128, cmap='viridis')
                        plt.ylabel("Frequency (Hz)")
                        plt.xlabel("Time (s)")
                        
                        # 2. Cross-Correlation Peaks
                        plt.subplot(2, 1, 2)
                        plt.title(f"2. FFT Spectrum | Target: {distance_m:.1f}m | SNR: {target_score:.1f} dB")
                        
                        freq_axis = np.linspace(0, sample_rate/2, len(correlation))
                        plt.plot(freq_axis, correlation, color='purple')

                        target_freq = freq_axis[target_idx] 

                        plt.axvline(x=0, color='red', linestyle='--', label=f'Leakage (0 Hz)')
                        plt.axvline(x=target_freq, color='green', linestyle='--', label=f'Beat Freq: {target_freq/1000:.1f} kHz')

                        plt.ylabel("Magnitude (Normalized)")
                        plt.xlabel("Beat Frequency (Hz)")

                        plt.xlim(-100, 10000) # zoom in
                        plt.legend(loc='upper right')

                        plt.tight_layout()
                        plt.savefig("latest_chirp.png")
                        plt.close()
                        
                        last_plot_time = current_time
                    # --------------------------------------

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DSP] {e}")

    except KeyboardInterrupt:
        pass

    print("[DSP] Process shutting down.")