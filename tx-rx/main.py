"""
main.py
~~~~~~~

This is the main class utilized to manage the SDR system. It will manage a main thread and 5 background threads, 
those being the transmit, receive, digital signal processing (DSP), flask server, and gimbal threads.
"""

import threading
import time
import queue 
import pandas as pd
import numpy as np
import bladerf as _bladerf
import json 
from trigger_ext import trigger_fire
from datetime import datetime
# from flask import Flask                 # for flask server + GUI communication
# from flask_socketio import SocketIO     # for flask server + GUI communication 

# External modules
from chirp_generator import gen_radar_chirp, ca_cfar_detect
from txrx import configure_sdr, process_radar_echo

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
        self.BANDWIDTH = data["bandwidth"]            
        self.CENTER_FREQ = data["center_frequency"]   
        self.NUM_RX_SAMPLES = data["num_rx_samples"]  
        self.PULSE_DURATION = data["pulse_duration"]  

        # Shared buffers for inter-thread communication
        self.rx_to_dsp_queue = queue.Queue(maxsize=10)  # Queue that holds received signals for DSP processing
        self.dsp_to_gui_queue = queue.Queue(maxsize=50) # Queue that holds processed data for GUI display

        # Hardware and data references
        self.sdr = None
        self.tx_bytes = None
        self.rx_bytes = None
        self.target_data = [] 

        # Background thread definitions
        self.threads = [
            threading.Thread(target=self._tx_bladerf, name="tx_thread"),
            threading.Thread(target=self._rx_bladerf, name="rx_thread"),
            threading.Thread(target=self._dsp, name="dsp_thread"),
            threading.Thread(target=self._flask_server, name="flask_thread"),
            threading.Thread(target=self._gimbal_control, name="gimbal_thread")
        ]

    def start(self):
        """ Intialize hardware, generate a chirp, and start background threads. """
        print("Initializing Radar System.")

        # Generating a 1ms LFM chirp 
        self.tx_bytes, self.tx_reference = gen_radar_chirp(self.SAMPLE_RATE, 
                                                           self.BANDWIDTH,
                                                           self.PULSE_DURATION)
        # Connect and configure bladeRF
        print("Connecting to bladeRF...")
        self.sdr = _bladerf.BladeRF()
        # Set transmit and receive channels to 
        self.tx_ch, self.rx_ch, self.tx_trig = configure_sdr(self.sdr, 
                                                             self.SAMPLE_RATE, 
                                                             self.CENTER_FREQ, 
                                                             self.BANDWIDTH)
                           
        print("BladeRF configured. Starting background threads.")
        
        # Starting the 5 background threads
        for thread in self.threads:
            thread.start()
        trigger_fire(self.sdr, self.tx_trig) # Fire once streaming the Tx and Rx channels are ready to go

        # Run the main class thread
        self._main_thread_logic()

    def _tx_bladerf(self):
        """ Continuously pushes the chirp out to the bladeRF. """
        while self.is_running:
            try: 
                tx_num_samples = len(self.tx_bytes) // 4 # 4 bytes per complex sample (I:int16 + Q:int16)
                self.sdr.sync_tx(self.tx_bytes, tx_num_samples, timeout_ms=1000) 
                # self.sdr.trigger_arm()
            except Exception as e:
                print(f"[Tx] {e}")

    def _rx_bladerf(self):
        """ Captures samples and sends to DSP queue. """
        rx_buf = bytearray(self.NUM_RX_SAMPLES * 4) # mult. by 4 to convert to standard 32-bit data types
        while self.is_running:
            try:
                # Capture echo signal
                self.sdr.sync_rx(rx_buf, self.NUM_RX_SAMPLES)

                # If not full, then add those bytes to the buffer
                if not self.rx_to_dsp_queue.full():
                    self.rx_to_dsp_queue.put(bytes(rx_buf))
            except Exception as e:
                print(f"[Rx] {e}")

    def _dsp(self):
        """ Digital Signal Processing thread logic. """
        while self.is_running:
            try:
                # Block until data is available in queue, timeout for 1 second for quick exit 
                rx_bytes = self.rx_to_dsp_queue.get(timeout=1)

                # Process the captured buffer cross-correlation score (time-delay) and distance
                correlation, distance, time_delay, rx_complex = process_radar_echo(rx_bytes, self.tx_reference, self.SAMPLE_RATE)

                # Apply CA-CFAR detection to the correlation output to find the peak and threshold
                peak_index, peak_value, threshold, is_detected, snr = ca_cfar_detect(correlation,ls guard_cells=9, ref_cells=32, pfa_factor=12.0)

                if is_detected:                
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                    hit_data = {
                        "Timestamp": timestamp,
                        "Correlation_Score": f"{peak_value:.1f}",
                        "Distance_m": f"{distance:.2f}",
                        "Time_Delay_ms": f"{time_delay:.1f}",
                        "Threshold": f"{threshold:.1f}",           # logging for now while tuning the pfa factor
                        "SNR_dB": f"{snr:.2f}" 
                    }                    
                    print(f"[DSP] ({timestamp}) Target detected. Range: {distance:.2f}m. "
                          f"Score: {peak_value:.1f} > Threshold: {threshold:.1f}. "
                          f"Peak Index: {peak_index}. " 
                          f"SNR: {snr:.1f} dB")
                     
                    # Store in main list for CSV logging and send to GUI
                    self.target_data.append(hit_data)
                    if not self.dsp_to_gui_queue.full():
                        self.dsp_to_gui_queue.put(hit_data)

                self.rx_to_dsp_queue.task_done()

            except queue.Empty:
                continue
            

    def _flask_server(self):
        """ Flask server thread logic. """
        while self.is_running:
            try:
                # Pull verified hits from the DSP queue
                gui_updates = self.dsp_to_gui_queue.get(timeout=1)
                # TODO: Add logic to send updates to the GUI via Flask-SocketIO
                # app = Flask(__name__)
                # socketio = SocketIO(app, cors_allowed_origins="*")

                # def send_update(data):
                #     socketio.emit('buffer_update', {'data': buffer_chunk})

                # socketio.run(app, port=5000)

            except queue.Empty:
                continue

    def _gimbal_control(self): 
        """ Gimbal control thread logic. """
        while self.is_running:

            # Want to keep gimbal in same position if the capture distance is greater than 50m.
            if self.target_data and float(self.target_data[-1]["Distance_m"]) < 50.0:
                print(f"[Gimbal] Target detected within 50m. Adjusting gimbal position...")
            else:
                print(f"[Gimbal] No target detected within 50m. Maintaining gimbal position.")
            
            time.sleep(1)

    # Main thread logic for the SDR system
    def _main_thread_logic(self):
        """ Main thread logic for the SDR system. """
        print("[Main] System is live. Press Ctrl+C to save data and exit.")

        # Run main control loop while background threads continue running
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupt received. Initializing shutdown.\n")
        finally:
            self.stop_system()

    def stop_system(self):
        """ Stops all threads and shuts down the SDR system. """
        threading.current_thread().name = "Main"
        print(f"\n[{threading.current_thread().name}] Shutting down system.")
        self.is_running = False # triggers the while loop to break
        
        # Waiting for all threads to finish
        for thread in self.threads:
            thread.join()
            print(f"[{threading.current_thread().name}] Thread {thread.name} has stopped.")

        # Save buffer to file 
        if self.target_data:
            df = pd.DataFrame(self.target_data)
            df.to_csv("radar_log.csv", index=False)
            print("Succesfully saved to radar_log.csv")

        # Disable SDR channels
        if self.sdr:
            self.tx_ch.enable = False
            self.rx_ch.enable = False
            print(f"[{threading.current_thread().name}] System shutdown complete.")

# Execution entry point
if __name__ == "__main__":
    system = Main()
    system.start()   