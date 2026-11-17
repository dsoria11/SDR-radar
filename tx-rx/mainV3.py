"""
mainV3.py
~~~~~~~

This is the main class utilized to manage the SDR system. It uses a per-chirp radar measurement loop startup modes:

Option 1: Skip calibration setup (keep metal sheet in front of the transmitter). 
    Load saved calibration_data.npz if available, or locks N_lat (hardware latency) on the first capture and runs without background subtraction.
Option 2: Run calibration setup (remove metal sheet from front of the transmitter first). 
    Records K=32 empty-scene captures, computes N_lat and coherent background b[n], and saves them to calibration_data.npz 
    for future Option 1 runs.

Using the following synchronization flags:
    - tx_chirp_prepped: True when the Tx thread has prepped a chirp and is ready to transmit.
    - rx_capturing: True when the Rx thread is actively capturing samples.
    - rx_buffer_full: True when the Rx thread has finished capturing samples and the buffer is full.
"""

import multiprocessing as mp
import threading
import argparse
import os
import time
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bladerf import _bladerf

# External modules
from flask_sdr_server import FlaskSDRServer
from chirp_generator import gen_radar_chirp
from dsp import *
from trigger_ext import * 

CALIB_FILE = "calibration_data.npz"

class Main:

    def __init__(self, mode=None):
        """ 
        Initializes the main class for the SDR system.
        
        Attributes:
            sample_rate (float): The sampling rate in Hz.
            bandwidth (float): The bandwidth of the chirp in Hz. [200KHz - 56MHz]
            center_frequency (float): The center frequency of the chirp in Hz.
            num_rx_samples (int): The number of Rx samples to capture for each radar echo. Has to be greater than:
                                  pulse_duration * sample_rate (i.e., the number of samples in the chirp).
            pulse_duration (float): The duration of the chirp in seconds.
        """
        with open("samples_config.json", "r") as file:
            self.cfg = json.load(file)

        # Run startup sanity checks
        check_config(self.cfg)

        self.SAMPLE_RATE = float(self.cfg["sample_rate"])
        self.CHIRP_BANDWIDTH = float(self.cfg["chirp_bandwidth"])      
        self.PULSE_DURATION = float(self.cfg["pulse_duration"])  
        self.SLOPE = self.CHIRP_BANDWIDTH / self.PULSE_DURATION

        self.CAPTURE_SAMPLES = int(self.cfg["capture_samples"]) # N_rx
        self.TX_OFFSET_SAMPLES = int(self.cfg["tx_offset_samples"]) # N_off 
        self.TX_AMPLITUDE = float(self.cfg["tx_amplitude"]) 
        self.PAD_FACTOR = int(self.cfg["zero_pad_factor"])
        self.MAX_RANGE = float(self.cfg["max_range_m"])
        self.BACKGROUND_CAPTURES = int(self.cfg["background_captures"])
        self.MEASUREMENT_INTERVAL = float(self.cfg["measurement_interval_s"]) 
        self.TIMEOUT = float(self.cfg["stream_timeout"]) # in ms

        self.mode = mode # mode 1: Skip calibration, mode 2: Run calibration

        # Boolean synchronization flags 
        self.state_lock = threading.Condition()
        self.tx_chirp_prepped = False 
        self.rx_capturing = False
        self.rx_buffer_full = False

        # Shutdown event
        self.stop_event = mp.Event()

        # Queues for Flask GUI process
        self.dsp_to_gui_queue = mp.Queue(maxsize=50) 
        self.iq_queue = mp.Queue(maxsize=5)   
        self.dsp_to_gui_queue.cancel_join_thread() 
        self.iq_queue.cancel_join_thread()

        # Hardware and data references
        self.sdr = None
        self.tx_bytes = None
        self.tx_frame = None
        self.chirp = None

        # Calibration state
        self.n_lat = None               # the hardware latency in samples (N_lat)
        self.complex_background = None
        self.target_data = []

    def _select_mode(self):
        """
        Prompts the user to select a startup mode for the SDR system.
        Mode 1: Skip calibration setup (keep metal sheet in front of the transmitter).
        Mode 2: Run calibration setup (remove metal sheet from front of the transmitter first).
        """
        if self.mode in (1,2):
            return 

        print("\n" + "=" * 68)
        print("Select startup mode:\n")
        print("1. Skip calibration setup (keep metal sheet in front of the transmitter).")
        print("2. Run calibration setup (remove metal sheet from front of the transmitter first).")
        print("=" * 68)

        while True:
            choice = input("Enter 1 or 2: ")
            if choice in ("", "1"):
                self.mode = 1
                break
            elif choice == "2":
                self.mode = 2
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def _prepare_calibration_state(self):
        """Initializes calibration variables based on Option 1 or Option 2."""
        if self.mode == 1:
            self.n_lat = None
            self.complex_background = np.zeros(self.chirp.size, dtype=np.complex64)
            print("\n[Option 1] Running fresh without calibration setup (metal sheet stays in place).")
            print("[Option 1] N_lat will lock onto the leakage peak on Cycle 1.\n")

        elif self.mode == 2:
            self.n_lat = None
            self.complex_background = None
            print(f"\n[Option 2] Remove metal sheet! Recording {self.BACKGROUND_CAPTURES} empty captures for:")
            print("  1. Trigger association test (time_zero_idx = N_off + N_lat)")
            print("  2. RX gain headroom check (keep peak < 0.5 / headroom >= 6 dB)")
            print("  3. Coherent leakage background b[n]\n")
            input("Press Enter once the metal sheet is removed to start calibration...")

    def start(self):
        """ 
        Intialize hardware, generate a chirp, and start background threads. 
        """
        print("Initializing Radar System.")

        self._select_mode()

        print("\nInitializing Radar System.")
        self.tx_bytes, self.chirp, self.tx_frame = gen_radar_chirp(
            self.SAMPLE_RATE,
            self.CHIRP_BANDWIDTH,
            self.PULSE_DURATION,
            n_rx=self.CAPTURE_SAMPLES,
            n_off=self.TX_OFFSET_SAMPLES,
            amplitude=self.TX_AMPLITUDE
        )

        self._prepare_calibration_state()

        # Connect and configure bladeRF
        print("Connecting to bladeRF...")
        self.sdr = _bladerf.BladeRF()
        self.tx_ch, self.rx_ch, self.tx_trig, self.rx_trig = configure_sdr(self.sdr, self.cfg)

        print("BladeRF configured. Starting Flask GUI process.")
        self.flask_process = mp.Process(target=self._flask_server, name="Flask", daemon=True)
        self.flask_process.start()

        self._main_loop()

    # def _capture_single_measurement(self, rx_buf):
    #     """
    #     Per measurement, arm triggers, enable channels, queue TX frame,
    #     fire hardware trigger, read N_rx samples, then disable channels.
    #     """
    #     trigger_arm(self.sdr, self.rx_trig, TriggerRole.SLAVE, arm=True)
    #     trigger_arm(self.sdr, self.tx_trig, TriggerRole.MASTER, arm=True)

    #     self.rx_ch.enable = True
    #     self.tx_ch.enable = True

    #     tx_err = []

    #     def tx_task():
    #         try:
    #             self.sdr.sync_tx(self.tx_bytes, self.n_rx, timeout_ms=self.timeout_ms)
    #         except Exception as e:
    #             tx_err.append(e)

    #     tx_thread = threading.Thread(target=tx_task, daemon=True)
    #     tx_thread.start()

    #     # Giving Python 1ms to enter sync_tx and stage the DMA transfer, before firing MINI_EXP_1
    #     def fire():
    #         time.sleep(1e-3)
    #         trigger_fire(self.sdr, self.tx_trig)

    #     fire_thread = threading.Thread(target=fire, daemon=True)
    #     fire_thread.start()

    #     self.sdr.sync_rx(rx_buf, self.n_rx, timeout_ms=self.timeout_ms)
    #     tx_thread.join(timeout=1.0)
    #     fire_thread.join(timeout=1.0)

    #     # Disrarm triggers before disabling channels so next cycle starts at a known sample point
    #     trigger_arm(self.sdr, self.rx_trig, TriggerRole.SLAVE, arm=False)
    #     trigger_arm(self.sdr, self.tx_trig, TriggerRole.MASTER, arm=False)

    #     self.tx_ch.enable = False
    #     self.rx_ch.enable = False

    #     if tx_err:
    #         raise RuntimeError(f"Tx: Encountered an error: {tx_err[0]}")

    def _tx_worker(self):
        """ 
        Synchronizes with the main loop using three boolean flags. 
        1. Sets tx_chirp_prepped flag to True, indicating that the chirp is ready for transmission.
        2. Waits for rx_capturing == True, then calls sync_tx to transmit the chirp.
        3. Waits for rx_buffer_full == True, resets tx_chirp_prepped flag = False, and waits for 
           rx_buffer_full == False before prepping the next cycle.
        """
        print("[Tx] Transmitter thread is live.")
        while not self.stop_event.is_set():
            with self.state_lock:
                # Prep chirp and notify Rx that Tx is ready
                self.tx_chirp_prepped = True
                self.state_lock.notify_all()

                # Wait for Rx to signal that capture has begun 
                while not self.rx_capturing and not self.stop_event.is_set():
                    self.state_lock.wait(timeout=0.5)
                if self.stop_event.is_set():
                    break

            try:
                # Queue the single chirp into the bladeRF Tx FIFO
                self.sdr.sync_tx(self.tx_bytes, self.CAPTURE_SAMPLES, timeout_ms=int(self.TIMEOUT))
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[Tx] {e}")

            with self.state_lock:
                # Wait for Rx buffer to fill up before prepping next chirp
                while not self.rx_buffer_full and not self.stop_event.is_set():
                    self.state_lock.wait(timeout=0.5)
                if self.stop_event.is_set():
                    break

                # Acknowledge that Tx completed and saw rx_buffer_full
                self.tx_chirp_prepped = False
                self.state_lock.notify_all()

                # Wait for main loop to finish DSP and reset rx_buffer_full flag to False
                while self.rx_buffer_full and not self.stop_event.is_set():
                    self.state_lock.wait(timeout=0.5)

    def _main_loop(self):
        print("[Main] System is live. Press Ctrl+C to save data and exit.")
        rx_buf = bytearray(self.CAPTURE_SAMPLES * 4)

        tx_thread = threading.Thread(target=self._tx_worker, daemon=True)
        tx_thread.start()

        calib_captures = []
        calib_latencies = []
        cycle = 0
        last_plot_time = 0.0

        try:
            while not self.stop_event.is_set():
                # --- STEP 1: Wait for tx_chirp_prepped == True, arm triggers & enable channels ---
                with self.state_lock:
                    while not self.tx_chirp_prepped and not self.stop_event.is_set():
                        self.state_lock.wait(timeout=0.5)
                    if self.stop_event.is_set():
                        break

                    # Reinitialize the sync interface 
                    self.sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1, fmt=_bladerf.Format.SC16_Q11,
                                        num_buffers=int(self.cfg["num_buffers"]), buffer_size=int(self.cfg["buffer_size"]), num_transfers=int(self.cfg["num_transfers"]), stream_timeout=int(self.cfg["stream_timeout"]))
                    self.sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1, fmt=_bladerf.Format.SC16_Q11,
                                        num_buffers=int(self.cfg["num_buffers"]), buffer_size=int(self.cfg["buffer_size"]), num_transfers=int(self.cfg["num_transfers"]), stream_timeout=int(self.cfg["stream_timeout"]))

                    # Arm triggers and enable channels fresh for this measurement
                    trigger_arm(self.sdr, self.rx_trig, TriggerRole.SLAVE, arm=True)
                    trigger_arm(self.sdr, self.tx_trig, TriggerRole.MASTER, arm=True)
                    self.rx_ch.enable = True
                    self.tx_ch.enable = True

                    self.rx_buffer_full = False
                    self.rx_capturing = True
                    self.state_lock.notify_all()

                # --- STEP 2: Fire hardware trigger once TX and RX are staged ---
                def fire_trigger():
                    time.sleep(1e-3)
                    trigger_fire(self.sdr, self.tx_trig)

                fire_thread = threading.Thread(target=fire_trigger, daemon=True)
                fire_thread.start()

                # Collect the fixed-length RX capture 
                self.sdr.sync_rx(rx_buf, self.CAPTURE_SAMPLES, timeout_ms=int(self.TIMEOUT))
                fire_thread.join(timeout=1.0)

                # --- STEP 3: Finish RX capture, disable channels, set rx_buffer_full = True ---
                with self.state_lock:
                    trigger_arm(self.sdr, self.rx_trig, TriggerRole.SLAVE, arm=False)
                    trigger_arm(self.sdr, self.tx_trig, TriggerRole.MASTER, arm=False)
                    self.tx_ch.enable = False
                    self.rx_ch.enable = False

                    self.rx_capturing = False
                    self.rx_buffer_full = True
                    self.state_lock.notify_all()

                    # Wait for _tx_worker to acknowledge rx_buffer_full (sets tx_chirp_prepped = False)
                    while self.tx_chirp_prepped and not self.stop_event.is_set():
                        self.state_lock.wait(timeout=0.5)

                cycle += 1

                # --- STEP 4 & 5: Unpack, verify association, dechirp, and calculate range ---
                rx, peak, headroom_db = unpack_sc16_q11(bytes(rx_buf))
                time_zero_idx, max_corr = find_leakage_index(rx, self.chirp)
                measured_lat = time_zero_idx - self.TX_OFFSET_SAMPLES

                # --- OPTION 2: Empty-Scene Calibration Setup ---
                if self.mode == 2 and self.complex_background is None:
                    calib_captures.append(rx)
                    calib_latencies.append(measured_lat)
                    clip_warn = " [WARNING: PEAK >= 0.5, LOWER GAIN!]" if peak >= 0.5 else ""
                    print(
                        f"[Calib {len(calib_captures):02d}/{self.BACKGROUND_CAPTURES:02d}] "
                        f"time_zero_idx={time_zero_idx} (N_lat={measured_lat:+d}) | "
                        f"peak={peak:.3f} ({headroom_db:.1f} dB headroom) | "
                        f"corr={max_corr:.1f}{clip_warn}"
                    )

                    if len(calib_captures) >= self.BACKGROUND_CAPTURES:
                        lat_std = float(np.std(calib_latencies))
                        self.n_lat = int(np.round(np.median(calib_latencies)))
                        fixed_start = self.TX_OFFSET_SAMPLES + self.n_lat

                        dechirped_stack = [
                            dechirp_at(r, self.chirp, fixed_start) for r in calib_captures
                        ]
                        self.complex_background = np.mean(dechirped_stack, axis=0)

                        print("\n" + "=" * 68)
                        print(f"[Association Summary] Median N_lat = {self.n_lat} samples (std = {lat_std:.2f} samples)")
                        print(f"[Fixed Dechirp Index] start = {self.TX_OFFSET_SAMPLES} + {self.n_lat} = {fixed_start}")
                        if lat_std > 2.0:
                            print("[WARNING] time_zero_idx wandered by > 2 samples! Check trigger sync.")
                        print("[Calibration Complete] Coherent background b[n] locked in memory.")
                        print("=" * 68)
                        input("\nPlace the metal sheet in front of the radar, then press Enter to start ranging...")
                        print()

                    # Release TX worker for the next cycle
                    with self.state_lock:
                        self.rx_buffer_full = False
                        self.state_lock.notify_all()

                    time.sleep(self.MEASUREMENT_INTERVAL)
                    continue

                # --- OPTION 1: Lock N_lat on Cycle 1 from leakage peak ---
                if self.n_lat is None:
                    self.n_lat = measured_lat
                    print(
                        f"[Option 1] Locked N_lat={self.n_lat:+d} on Cycle 1 "
                        f"(fixed_start={self.TX_OFFSET_SAMPLES + self.n_lat}, "
                        f"peak={peak:.3f}, headroom={headroom_db:.1f} dB)."
                    )

                fixed_start = self.TX_OFFSET_SAMPLES + self.n_lat

                if 0 <= fixed_start and (fixed_start + self.chirp.size) <= rx.size:
                    mixed_raw = dechirp_at(rx, self.chirp, fixed_start)
                    mixed_clean = mixed_raw - self.complex_background

                    ranges, spec = range_spectrum(mixed_clean, self.SAMPLE_RATE, self.SLOPE, self.PAD_FACTOR)
                    distance_m, peak_bin, peak_val = pick_peak(ranges, spec, self.MAX_RANGE)

                    k_max = np.searchsorted(ranges, self.MAX_RANGE)
                    noise_floor = float(np.median(spec[1:k_max])) + 1e-12
                    snr_db = 20.0 * np.log10((peak_val + 1e-12) / noise_floor)
                    beat_freq_hz = (2.0 * self.SLOPE * distance_m) / 3e8

                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                    print(
                        f"[Cycle {cycle:04d}] ({timestamp}) "
                        f"time_zero_idx={time_zero_idx} | peak={peak:.2f} | "
                        f"Beat={beat_freq_hz:6.1f} Hz | Range={distance_m:5.2f} m | SNR={snr_db:5.1f} dB"
                    )

                    # --- STEP 6: Send capture and result to Flask GUI ---
                    rx_seg = rx[fixed_start : fixed_start + self.chirp.size]
                    if not self.iq_queue.full():
                        self.iq_queue.put(rx_seg)

                    hit_data = {
                        "Timestamp": timestamp,
                        "Correlation_Score": round(snr_db, 2),
                        "Distance_m": round(distance_m, 3),
                        "Time_Delay_ms": round((2.0 * distance_m / 3e8) * 1e6, 4)
                    }
                    self.target_data.append(hit_data)
                    if not self.dsp_to_gui_queue.full():
                        self.dsp_to_gui_queue.put(hit_data)

                    # Save diagnostic plot once per second 
                    now = time.time()
                    if now - last_plot_time > 1.0:
                        _, spec_unsubtracted = range_spectrum(mixed_raw, self.SAMPLE_RATE, self.SLOPE, self.PAD_FACTOR)
                        plt.figure(figsize=(10, 8))

                        plt.subplot(2, 1, 1)
                        plt.title(f"1. |rx| Capture Envelope (peak={peak:.2f}, start={fixed_start})")
                        plt.plot(np.abs(rx), color="steelblue", linewidth=0.7)
                        plt.axvline(fixed_start, color="red", linestyle="--", label="N_off + N_lat (Start)")
                        plt.axvline(fixed_start + self.chirp.size, color="orange", linestyle="--", label="Chirp End")
                        plt.xlabel("Sample Index")
                        plt.ylabel("|rx| (Normalized to 2048)")
                        plt.legend(loc="upper right")

                        plt.subplot(2, 1, 2)
                        plt.title(f"2. Range Spectrum | Peak: {distance_m:.2f} m | SNR: {snr_db:.1f} dB")
                        plt.plot(ranges[:k_max], spec_unsubtracted[:k_max], color="gray", alpha=0.6, label="Raw Dechirp (Leakage)")
                        plt.plot(ranges[:k_max], spec[:k_max], color="purple", linewidth=1.5, label="Processed Spectrum")
                        plt.axvline(distance_m, color="green", linestyle="--", label=f"Target: {distance_m:.2f} m")
                        plt.xlabel("Range (m)")
                        plt.ylabel("FFT Magnitude")
                        plt.xlim(0, self.MAX_RANGE)
                        plt.legend(loc="upper right")

                        plt.tight_layout()
                        plt.savefig("latest_chirp.png")
                        plt.close()
                        last_plot_time = now
                else:
                    print(f"[Cycle {cycle:04d}] Warning: fixed_start={fixed_start} out of bounds.")

                # --- STEP 7: Reset rx_buffer_full = False to release _tx_worker for next cycle ---
                with self.state_lock:
                    self.rx_buffer_full = False
                    self.state_lock.notify_all()

                time.sleep(self.MEASUREMENT_INTERVAL)

        except KeyboardInterrupt:
            print("\nInterrupt Received. Shutting down now...")
        finally:
            self.stop_system()

    def _flask_server(self):
        """ Flask server thread logic """
        flask_server = FlaskSDRServer()
        flask_server.update_tx_reference(self.tx_reference)
        flask_server.run_server(
            dsp_queue=self.dsp_to_gui_queue,
            iq_queue=self.iq_queue,
            stop_event=self.stop_event
        )

    def _gimbal_control(self):
        """ Gimbal control thread logic. """
        try:
            while not self.stop_event.is_set():
                try:
                    gui_updates = self.dsp_to_gui_queue.get(timeout=1)
                except queue.Empty:
                    continue
        
        except KeyboardInterrupt:
            pass

    def stop_system(self):
        """ Stops all processes and shuts down the SDR system. """
        print("\n[Main] Shutting down system.")
        self.stop_event.set()
        
        with self.state_lock:
            self.state_lock.notify_all()

        if hasattr(self, "flask_process") and self.flask_process.is_alive():
            self.flask_process.join(timeout=1)
            print("[Main] Flask process has stopped.")

        if len(self.target_data) > 0:
            df = pd.DataFrame(self.target_data)
            df.to_csv("radar_log.csv", index=False)
            print("Successfully saved to radar_log.csv")
        
        if self.sdr:
            try:
                self.tx_ch.enable = False
                self.rx_ch.enable = False
            except Exception as e:
                pass
            print("[Main] Hardware disabled. System shutdown complete.")

# Execution entry point
if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Per-Chirp FMCW Radar")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1,2],
        default=None,
        help="1 = Do not run calibration (leave target in place), 2 = Run calibration (remove target)"
    )
    args = parser.parse_args()
    system = Main(mode=args.mode)
    system.start()

# """
# mainV3.py
# ~~~~~~~~~
# Per-chirp radar measurement loop using boolean synchronization flags.

# Startup Modes:
#   Option 1: Do NOT run calibration setup (leave metal sheet in place).
#   Option 2: Run calibration setup (remove metal sheet first). Computes
#             a Magnitude-Domain background profile to bypass SDR phase incoherence.
# """

# import argparse
# import os
# import multiprocessing as mp
# import threading
# import time
# import json
# import queue
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from bladerf import _bladerf

# from flask_sdr_server import FlaskSDRServer
# from chirp_generator import gen_radar_chirp
# from dsp import (
#     check_config,
#     configure_sdr,
#     unpack_sc16_q11,
#     find_leakage_index,
#     dechirp_at,
#     range_spectrum,
#     pick_peak
# )
# from trigger_ext import trigger_arm, trigger_fire, TriggerRole

# CALIB_FILE = "calibration_data.npz"


# class Main:
#     def __init__(self, mode=None):
#         with open("samples_config.json", "r") as file:
#             self.cfg = json.load(file)

#         check_config(self.cfg)

#         self.SAMPLE_RATE = float(self.cfg["sample_rate"])
#         self.CHIRP_BANDWIDTH = float(self.cfg["chirp_bandwidth"])      
#         self.PULSE_DURATION = float(self.cfg["pulse_duration"])  
#         self.SLOPE = self.CHIRP_BANDWIDTH / self.PULSE_DURATION

#         self.CAPTURE_SAMPLES = int(self.cfg["capture_samples"]) 
#         self.TX_OFFSET_SAMPLES = int(self.cfg["tx_offset_samples"]) 
#         self.TX_AMPLITUDE = float(self.cfg["tx_amplitude"]) 
#         self.PAD_FACTOR = int(self.cfg["zero_pad_factor"])
#         self.MAX_RANGE = float(self.cfg["max_range_m"])
#         self.BACKGROUND_CAPTURES = int(self.cfg["background_captures"])
#         self.MEASUREMENT_INTERVAL = float(self.cfg["measurement_interval_s"]) 
#         self.TIMEOUT = float(self.cfg["stream_timeout"])

#         self.mode = mode 

#         self.state_lock = threading.Condition()
#         self.tx_chirp_prepped = False 
#         self.rx_capturing = False
#         self.rx_buffer_full = False

#         self.stop_event = mp.Event()
#         self.dsp_to_gui_queue = mp.Queue(maxsize=50) 
#         self.iq_queue = mp.Queue(maxsize=5)   
#         self.dsp_to_gui_queue.cancel_join_thread() 
#         self.iq_queue.cancel_join_thread()

#         self.sdr = None
#         self.tx_bytes = None
#         self.tx_frame = None
#         self.chirp = None

#         # Calibration state
#         self.n_lat = None               
#         self.mag_background = None  # Using Magnitude background instead of Complex
#         self.target_data = []

#     def select_mode(self):
#         if self.mode in (1,2):
#             return 

#         print("\n" + "=" * 68)
#         print("Select startup mode:\n")
#         print("1. Skip calibration setup (keep metal sheet in front of the transmitter).")
#         print("2. Run calibration setup (remove metal sheet from front of the transmitter first).")
#         print("=" * 68)

#         while True:
#             choice = input("Enter 1 or 2 [default=1]: ").strip()
#             if choice in ("", "1"):
#                 self.mode = 1
#                 break
#             elif choice == "2":
#                 self.mode = 2
#                 break
#             else:
#                 print("Invalid choice. Please enter 1 or 2.")

#     def _prepare_calibration_state(self):
#         if self.mode == 1:
#             if os.path.exists(CALIB_FILE):
#                 try:
#                     data = np.load(CALIB_FILE)
#                     if "mag_background" in data:
#                         self.mag_background = data["mag_background"]
#                         self.n_lat = int(data["n_lat"])
#                         print(f"\n[Option 1] Loaded saved calibration from '{CALIB_FILE}' (N_lat={self.n_lat}).")
#                         print("[Option 1] Metal sheet can stay in place. Starting radar...\n")
#                         return
#                 except Exception as e:
#                     print(f"[Option 1] Could not load '{CALIB_FILE}' ({e}). Running uncalibrated.")

#             self.n_lat = None
#             self.mag_background = None
#             print("\n[Option 1] Running fresh without calibration setup.")
#             print("[Option 1] N_lat will lock onto the leakage peak on Cycle 1.\n")

#         elif self.mode == 2:
#             self.n_lat = None
#             self.mag_background = None
#             print(f"\n[Option 2] Remove metal sheet! Recording {self.BACKGROUND_CAPTURES} empty captures for:")
#             print("  1. Trigger association test (time_zero_idx = N_off + N_lat)")
#             print("  2. Magnitude-Domain Leakage Background (Phase-blind subtraction)\n")
#             input("Press Enter once the metal sheet is removed to start calibration...")

#     def start(self):
#         self.select_mode()

#         print("\nInitializing Radar System.")
#         self.tx_bytes, self.chirp, self.tx_frame = gen_radar_chirp(
#             self.SAMPLE_RATE,
#             self.CHIRP_BANDWIDTH,
#             self.PULSE_DURATION,
#             n_rx=self.CAPTURE_SAMPLES,
#             n_off=self.TX_OFFSET_SAMPLES,
#             amplitude=self.TX_AMPLITUDE
#         )

#         self._prepare_calibration_state()

#         print("Connecting to bladeRF...")
#         self.sdr = _bladerf.BladeRF()
#         self.tx_ch, self.rx_ch, self.tx_trig, self.rx_trig = configure_sdr(self.sdr, self.cfg)

#         print("BladeRF configured. Starting Flask GUI process.")
#         self.flask_process = mp.Process(target=self._flask_server, name="Flask", daemon=True)
#         self.flask_process.start()

#         self._main_loop()

#     def _tx_worker(self):
#         print("[Tx] Transmitter thread is live.")
#         while not self.stop_event.is_set():
#             with self.state_lock:
#                 self.tx_chirp_prepped = True
#                 self.state_lock.notify_all()

#                 while not self.rx_capturing and not self.stop_event.is_set():
#                     self.state_lock.wait(timeout=0.5)
#                 if self.stop_event.is_set():
#                     break

#             try:
#                 self.sdr.sync_tx(self.tx_bytes, self.CAPTURE_SAMPLES, timeout_ms=int(self.TIMEOUT))
#             except Exception as e:
#                 if not self.stop_event.is_set():
#                     print(f"[Tx Error] {e}")

#             with self.state_lock:
#                 while not self.rx_buffer_full and not self.stop_event.is_set():
#                     self.state_lock.wait(timeout=0.5)
#                 if self.stop_event.is_set():
#                     break

#                 self.tx_chirp_prepped = False
#                 self.state_lock.notify_all()

#                 while self.rx_buffer_full and not self.stop_event.is_set():
#                     self.state_lock.wait(timeout=0.5)

#     def _main_loop(self):
#         print("[Main] System is live. Press Ctrl+C to save data and exit.")
#         rx_buf = bytearray(self.CAPTURE_SAMPLES * 4)

#         tx_thread = threading.Thread(target=self._tx_worker, daemon=True)
#         tx_thread.start()

#         calib_captures = []
#         calib_latencies = []
#         cycle = 0
#         last_plot_time = 0.0

#         try:
#             while not self.stop_event.is_set():
#                 with self.state_lock:
#                     while not self.tx_chirp_prepped and not self.stop_event.is_set():
#                         self.state_lock.wait(timeout=0.5)
#                     if self.stop_event.is_set():
#                         break

#                     # Re-initialize sync buffers to prevent bladeRF state corruption
#                     self.sdr.sync_config(
#                         layout=_bladerf.ChannelLayout.TX_X1,
#                         fmt=_bladerf.Format.SC16_Q11,
#                         num_buffers=int(self.cfg["num_buffers"]),
#                         buffer_size=int(self.cfg["buffer_size"]),
#                         num_transfers=int(self.cfg["num_transfers"]),
#                         stream_timeout=int(self.TIMEOUT)
#                     )
#                     self.sdr.sync_config(
#                         layout=_bladerf.ChannelLayout.RX_X1,
#                         fmt=_bladerf.Format.SC16_Q11,
#                         num_buffers=int(self.cfg["num_buffers"]),
#                         buffer_size=int(self.cfg["buffer_size"]),
#                         num_transfers=int(self.cfg["num_transfers"]),
#                         stream_timeout=int(self.TIMEOUT)
#                     )

#                     trigger_arm(self.sdr, self.rx_trig, TriggerRole.SLAVE, arm=True)
#                     trigger_arm(self.sdr, self.tx_trig, TriggerRole.MASTER, arm=True)
#                     self.rx_ch.enable = True
#                     self.tx_ch.enable = True

#                     self.rx_buffer_full = False
#                     self.rx_capturing = True
#                     self.state_lock.notify_all()

#                 def fire_trigger():
#                     time.sleep(1e-3)
#                     trigger_fire(self.sdr, self.tx_trig)

#                 fire_thread = threading.Thread(target=fire_trigger, daemon=True)
#                 fire_thread.start()

#                 self.sdr.sync_rx(rx_buf, self.CAPTURE_SAMPLES, timeout_ms=int(self.TIMEOUT))
#                 fire_thread.join(timeout=1.0)

#                 with self.state_lock:
#                     trigger_arm(self.sdr, self.rx_trig, TriggerRole.SLAVE, arm=False)
#                     trigger_arm(self.sdr, self.tx_trig, TriggerRole.MASTER, arm=False)
#                     self.tx_ch.enable = False
#                     self.rx_ch.enable = False

#                     self.rx_capturing = False
#                     self.rx_buffer_full = True
#                     self.state_lock.notify_all()

#                     while self.tx_chirp_prepped and not self.stop_event.is_set():
#                         self.state_lock.wait(timeout=0.5)

#                 cycle += 1

#                 rx, peak, headroom_db = unpack_sc16_q11(bytes(rx_buf))
#                 time_zero_idx, max_corr = find_leakage_index(rx, self.chirp)
#                 measured_lat = time_zero_idx - self.TX_OFFSET_SAMPLES

#                 # --- OPTION 2: Empty-Scene Calibration Setup (Magnitude Domain) ---
#                 if self.mode == 2 and self.mag_background is None:
                    
#                     # 1. Dechirp and remove DC offset to minimize the native leakage hump
#                     mixed_raw = dechirp_at(rx, self.chirp, self.TX_OFFSET_SAMPLES + measured_lat)
#                     mixed_raw -= np.mean(mixed_raw)
                    
#                     # 2. Get the FFT Magnitude spectrum
#                     ranges, spec = range_spectrum(mixed_raw, self.SAMPLE_RATE, self.SLOPE, self.PAD_FACTOR)
                    
#                     calib_captures.append(spec)
#                     calib_latencies.append(measured_lat)
                    
#                     clip_warn = " [WARNING: PEAK >= 0.5, LOWER GAIN!]" if peak >= 0.5 else ""
#                     print(
#                         f"[Calib {len(calib_captures):02d}/{self.BACKGROUND_CAPTURES:02d}] "
#                         f"time_zero_idx={time_zero_idx} (N_lat={measured_lat:+d}) | "
#                         f"peak={peak:.3f} | corr={max_corr:.1f}{clip_warn}"
#                     )

#                     if len(calib_captures) >= self.BACKGROUND_CAPTURES:
#                         lat_std = float(np.std(calib_latencies))
#                         self.n_lat = int(np.round(np.median(calib_latencies)))
                        
#                         # 3. Average the MAGNITUDES (Bypasses SDR Phase Incoherence)
#                         self.mag_background = np.mean(calib_captures, axis=0)

#                         np.savez(
#                             CALIB_FILE,
#                             n_lat=self.n_lat,
#                             mag_background=self.mag_background
#                         )

#                         print("\n" + "=" * 68)
#                         print(f"[Association Summary] Median N_lat = {self.n_lat} samples (std = {lat_std:.2f} samples)")
#                         if lat_std > 2.0:
#                             print("[WARNING] time_zero_idx wandered by > 2 samples! Check trigger sync.")
#                         print("[Calibration Complete] Magnitude background locked in memory.")
#                         print("=" * 68)
#                         input("\nPlace the metal sheet in front of the radar, then press Enter to start ranging...")
#                         print()

#                     with self.state_lock:
#                         self.rx_buffer_full = False
#                         self.state_lock.notify_all()

#                     time.sleep(self.MEASUREMENT_INTERVAL)
#                     continue

#                 if self.n_lat is None:
#                     self.n_lat = measured_lat
#                     print(f"[Option 1] Locked N_lat={self.n_lat:+d} on Cycle 1.")

#                 fixed_start = self.TX_OFFSET_SAMPLES + self.n_lat

#                 if 0 <= fixed_start and (fixed_start + self.chirp.size) <= rx.size:
#                     # 1. Dechirp and remove DC spike
#                     mixed_raw = dechirp_at(rx, self.chirp, fixed_start)
#                     mixed_raw -= np.mean(mixed_raw)

#                     # 2. Get Raw Spectrum
#                     ranges, spec_raw = range_spectrum(mixed_raw, self.SAMPLE_RATE, self.SLOPE, self.PAD_FACTOR)
                    
#                     # 3. Subtract Magnitude Background (if available)
#                     if self.mag_background is not None:
#                         spec_clean = np.maximum(spec_raw - self.mag_background, 0.0)
#                     else:
#                         spec_clean = spec_raw

#                     # 4. Find Target
#                     distance_m, peak_bin, peak_val = pick_peak(ranges, spec_clean, self.MAX_RANGE)

#                     k_max = np.searchsorted(ranges, self.MAX_RANGE)
#                     noise_floor = float(np.median(spec_clean[1:k_max])) + 1e-12
#                     snr_db = 20.0 * np.log10((peak_val + 1e-12) / noise_floor)
#                     beat_freq_hz = (2.0 * self.SLOPE * distance_m) / 3e8

#                     timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]
#                     print(
#                         f"[Cycle {cycle:04d}] ({timestamp}) "
#                         f"time_zero_idx={time_zero_idx} | peak={peak:.2f} | "
#                         f"Beat={beat_freq_hz:6.1f} Hz | Range={distance_m:5.2f} m | SNR={snr_db:5.1f} dB"
#                     )

#                     rx_seg = rx[fixed_start : fixed_start + self.chirp.size]
#                     if not self.iq_queue.full():
#                         self.iq_queue.put(rx_seg)

#                     hit_data = {
#                         "Timestamp": timestamp,
#                         "Correlation_Score": round(snr_db, 2),
#                         "Distance_m": round(distance_m, 3),
#                         "Time_Delay_ms": round((2.0 * distance_m / 3e8) * 1e6, 4)
#                     }
#                     self.target_data.append(hit_data)
#                     if not self.dsp_to_gui_queue.full():
#                         self.dsp_to_gui_queue.put(hit_data)

#                     now = time.time()
#                     if now - last_plot_time > 1.0:
#                         plt.figure(figsize=(10, 8))

#                         plt.subplot(2, 1, 1)
#                         plt.title(f"1. |rx| Capture Envelope (peak={peak:.2f}, start={fixed_start})")
#                         plt.plot(np.abs(rx), color="steelblue", linewidth=0.7)
#                         plt.axvline(fixed_start, color="red", linestyle="--", label="N_off + N_lat (Start)")
#                         plt.axvline(fixed_start + self.chirp.size, color="orange", linestyle="--", label="Chirp End")
#                         plt.xlabel("Sample Index")
#                         plt.ylabel("|rx| (Normalized to 2048)")
#                         plt.legend(loc="upper right")

#                         plt.subplot(2, 1, 2)
#                         plt.title(f"2. Range Spectrum | Peak: {distance_m:.2f} m | SNR: {snr_db:.1f} dB")
#                         plt.plot(ranges[:k_max], spec_raw[:k_max], color="gray", alpha=0.6, label="Raw Dechirp (Leakage)")
#                         plt.plot(ranges[:k_max], spec_clean[:k_max], color="purple", linewidth=1.5, label="Processed Spectrum")
#                         plt.axvline(distance_m, color="green", linestyle="--", label=f"Target: {distance_m:.2f} m")
#                         plt.xlabel("Range (m)")
#                         plt.ylabel("FFT Magnitude")
#                         plt.xlim(0, self.MAX_RANGE)
#                         plt.legend(loc="upper right")

#                         plt.tight_layout()
#                         plt.savefig("latest_chirp.png")
#                         plt.close()
#                         last_plot_time = now
#                 else:
#                     print(f"[Cycle {cycle:04d}] Warning: fixed_start={fixed_start} out of bounds.")

#                 with self.state_lock:
#                     self.rx_buffer_full = False
#                     self.state_lock.notify_all()

#                 time.sleep(self.MEASUREMENT_INTERVAL)

#         except KeyboardInterrupt:
#             print("\nInterrupt Received. Shutting down now...")
#         finally:
#             self.stop_system()

#     def _flask_server(self):
#         flask_server = FlaskSDRServer()
#         flask_server.update_tx_reference(self.chirp)
#         flask_server.run_server(
#             dsp_queue=self.dsp_to_gui_queue,
#             iq_queue=self.iq_queue,
#             stop_event=self.stop_event
#         )

#     def _gimbal_control(self):
#         try:
#             while not self.stop_event.is_set():
#                 try:
#                     gui_updates = self.dsp_to_gui_queue.get(timeout=1)
#                 except queue.Empty:
#                     continue
#         except KeyboardInterrupt:
#             pass

#     def stop_system(self):
#         print("\n[Main] Shutting down system.")
#         self.stop_event.set()
        
#         with self.state_lock:
#             self.state_lock.notify_all()

#         if hasattr(self, "flask_process") and self.flask_process.is_alive():
#             self.flask_process.join(timeout=1)
#             print("[Main] Flask process has stopped.")

#         if len(self.target_data) > 0:
#             df = pd.DataFrame(self.target_data)
#             df.to_csv("radar_log.csv", index=False)
#             print("Successfully saved to radar_log.csv")
        
#         if self.sdr:
#             try:
#                 self.tx_ch.enable = False
#                 self.rx_ch.enable = False
#             except Exception:
#                 pass
#             print("[Main] Hardware disabled. System shutdown complete.")

# if __name__ == "__main__":
#     mp.freeze_support()
#     parser = argparse.ArgumentParser(description="Per-Chirp FMCW Radar")
#     parser.add_argument(
#         "--mode",
#         type=int,
#         choices=[1,2],
#         default=None,
#         help="1 = Do not run calibration (leave target in place), 2 = Run calibration (remove target)"
#     )
#     args = parser.parse_args()
    
#     system = Main(mode=args.mode)
#     system.start()