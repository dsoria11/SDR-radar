"""
mainV2.py
~~~~~~~

This is the main class utilized to manage the SDR system. It will manage two tx and rx threads, and 4 processes  
with those being the digital signal processing (DSP), flask server, and gimbal threads.

Here we are continuously receiving and transmitting intermittently.

4 multiprocessing processes:
    1. Main Process: Splits into two threads for simultaneous I/O paths (tx loop and rx worker)
    2. DSP Process
    3. Flask Process
    4. Gimbal Process

"""

import multiprocessing as mp
import time
import pandas as pd
import numpy as np
import bladerf as _bladerf
import json 
import queue
import threading
from multiprocessing import Manager, Process, Queue 
from datetime import datetime

# External modules
from chirp_generator import gen_radar_chirp, ca_cfar_detect
from dsp import configure_sdr, process_radar_echo, run_dsp_process

class Main:
    def __init__(self):
        """ Initializes the main class for the SDR system.
        
        Attributes:
            sample_rate (float): The sampling rate in Hz. [521KHz - 61.44MHz]
            bandwidth (float): The bandwidth of the chirp in Hz. [200KHz - 56MHz]
            center_frequency (float): The center frequency of the chirp in Hz.
            num_rx_samples (int): The number of Rx samples to capture for each radar echo. Has to be greater than:
                                  pulse_duration * sample_rate (i.e., the number of samples in the chirp).
            pulse_duration (float): The duration of the chirp in seconds.
        """
        self.is_running = True

        try:
            with open("samples_config.json", "r") as file:
                data = json.load(file)
        except FileNotFoundError:
                print("Error: samples_config.json not found.")

        self.SAMPLE_RATE = data["sample_rate"]  
        self.RF_BANDWIDTH = data["rf_bandwidth"]
        self.CHIRP_BANDWIDTH = data["chirp_bandwidth"]      
        self.CENTER_FREQ = data["center_frequency"]   
        self.NUM_RX_SAMPLES = data["num_rx_samples"]  
        self.PULSE_DURATION = data["pulse_duration"]  

        # Set synchronziation primitives 
        self.stop_event = mp.Event()

        # Inter-process communication queues (IPC)
        self.rx_to_dsp_queue = mp.Queue(maxsize=10)  # Queue that holds received signals for DSP processing
        self.dsp_to_gui_queue = mp.Queue(maxsize=50) # Queue that holds processed data for GUI display

        self.rx_to_dsp_queue.cancel_join_thread()
        self.dsp_to_gui_queue.cancel_join_thread()

        # Shared memory manager for IPC data logging
        self.manager = mp.Manager()
        self.target_data = self.manager.list()           

        # Hardware and data references
        self.sdr = None
        self.tx_bytes = None
        self.rx_bytes = None

    def start(self):
        """ Intialize hardware, generate a chirp, and start background threads. """
        print("Initializing Radar System.")

        # Generating a LFM chirp 
        self.tx_bytes, self.tx_reference = gen_radar_chirp(self.SAMPLE_RATE, 
                                                           self.CHIRP_BANDWIDTH,
                                                           self.PULSE_DURATION,
                                                           frame_samples=self.NUM_RX_SAMPLES)
        # Connect and configure bladeRF
        print("Connecting to bladeRF...")
        self.sdr = _bladerf.BladeRF()
        self.tx_ch, self.rx_ch = configure_sdr(self.sdr, 
                                               self.SAMPLE_RATE, 
                                               self.CENTER_FREQ, 
                                               self.RF_BANDWIDTH)
                           
        print("BladeRF configured. Starting background threads.")
        
        # Define the isolated background processes 
        self.processes = [
            mp.Process(target=run_dsp_process, name="DSP", args=(
                self.stop_event,        # syncs shutdown
                self.rx_to_dsp_queue,   # receive raw bytes
                self.dsp_to_gui_queue,  # send hits to GUI (flask server)
                self.target_data,       # to log to CSV
                self.tx_reference,      # array for correlation
                self.SAMPLE_RATE      
            )),
            mp.Process(target=self._flask_server, name="Flask"),
            mp.Process(target=self._gimbal_control, name="Gimbal"),
        ]

        # Start the processes
        for p in self.processes:
            p.daemon = True
            p.start()

        self._main_io_loop()

    def _tx_worker(self, tx_num_samples):
        """ Runs intermittently in the background """
        print("[Tx] Transmitter thread is live.")
        while not self.stop_event.is_set():
            try:
                self.sdr.sync_tx(self.tx_bytes, tx_num_samples, timeout_ms=1000)
                time.sleep(0.5) # wait 0.5 seconds in between chirps
            except Exception as e:
                if not self.stop_event.is_set():
                    pass

    # def _rx_worker(self, rx_buf):
    #     """ Adding received samples to buffer """
    #     try:
    #         self.sdr.sync_rx(rx_buf, self.NUM_RX_SAMPLES)
    #         if not self.rx_to_dsp_queue.full():
    #             self.rx_to_dsp_queue.put(bytes(rx_buf))
    #     except Exception as e:
    #         print(f"[Rx Error] {e}")

    def _main_io_loop(self):
        """ Handles continous received data while Tx pulses independently """
        print("[Main] System is live. Press Ctrl+C to save data and exit.")
        rx_buf = bytearray(self.NUM_RX_SAMPLES * 4)
        tx_num_samples = len(self.tx_bytes) // 4

        # start the Tx background thread
        tx_thread = threading.Thread(target=self._tx_worker, args=(tx_num_samples,))
        tx_thread.daemon = True
        tx_thread.start()

        try:
            # Capture Rx buffers constantly
            while not self.stop_event.is_set():
                try:
                    self.sdr.sync_rx(rx_buf, self.NUM_RX_SAMPLES)
                    if not self.rx_to_dsp_queue.full():
                        self.rx_to_dsp_queue.put(bytes(rx_buf))
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"[Rx] {e}")

        except KeyboardInterrupt:
            print("\nInterrupt Received. Shutting down now\n")
        finally:
            self.stop_system()

    def _flask_server(self):
        """ Flask server thread logic. """
        try:
            while not self.stop_event.is_set():
                try:
                    gui_updates = self.dsp_to_gui_queue.get(timeout=1)

                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            pass

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
        """ Stops all threads and shuts down the SDR system. """
        print("\n[Main] Shutting down system.")
        self.stop_event.set()
        
        # Join background processes
        for p in self.processes:
            p.join(timeout=1)
            print(f"[Main] Process {p.name} has stopped.")

        if len(self.target_data) > 0:
            # Convert managed list to standard list for pandas
            df = pd.DataFrame(list(self.target_data))
            df.to_csv("radar_log.csv", index=False)
            print("Succesfully saved to radar_log.csv")

        # Disable SDR channels
        if self.sdr:
            self.tx_ch.enable = False
            self.rx_ch.enable = False
            print(f"[Main] Hardware disabled. System shutdown complete.")

# Execution entry point
if __name__ == "__main__":
    mp.freeze_support() 
    system = Main()
    system.start()  