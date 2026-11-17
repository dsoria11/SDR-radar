"""
flask_sdr_server.py
~~~~~~~~
Takes data from DSP.py and creates a Flask API for accessing the latest IQ 
samples, distance measurements, and FFT frames.
"""

import json
import threading
import numpy as np
import time
import queue

from flask import Flask, jsonify
from flask_cors import CORS
 
# Muting the HTTP logs
import logging 
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

class FlaskSDRServer:

    def __init__(self):

        try:
            with open("samples_config.json", "r") as file:
                data = json.load(file)

        except FileNotFoundError:
            raise FileNotFoundError(
                "samples_config.json not found."
            )

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid samples_config.json: {e}"
            )

        self.sample_rate = data["sample_rate"]
        self.fft_size = data["fft_size"]

        self.latest_iq = None
        self.latest_iq_timestamp = None
        self.iq_lock = threading.Lock()

        self.initial_distance_time = None
        self.latest_distance = None
        self.latest_distance_time = None
        self.latest_confidence = None
        self.distance_lock = threading.Lock()

        self.tx_reference = None

        self.app = Flask(__name__)
        CORS(self.app)

        self._register_routes()

    def consume_dsp_queue(self, dsp_queue, stop_event):
        """
        Receive DSP updates from the multiprocessing queue.
        """

        while not stop_event.is_set():
            try:
                hit_data = dsp_queue.get(timeout=1)

                # if message["type"] == "detection":
                #    self.update_detection(message["data"])
                if "Distance_m" in hit_data:
                    self.update_detection(hit_data)
            except queue.Empty:
                continue

    def update_live_iq(self, iq_samples):
        """
        Receive IQ samples from DSP.py.
        """

        iq_array = np.asarray(iq_samples, dtype=np.complex64)

        with self.iq_lock:
            self.latest_iq = iq_array.copy()
            self.latest_iq_timestamp = time.time()

    def consume_iq_queue(self, iq_queue, stop_event):
        """ 
        Receive IQ samples from the multiprocessing queue.
        """
        while not stop_event.is_set():
            try:
                iq_samples = iq_queue.get(timeout=1)
                self.update_live_iq(iq_samples)
            except queue.Empty:
                continue

    def update_tx_reference(self, tx_reference):
        """
        Store the transmitted reference chirp.
        """
        self.tx_reference = np.asarray(
            tx_reference,
            dtype=np.complex64
        ).copy()

    def update_detection(self, hit_data):
        """
        Receive the latest target detection from DSP.py.
        """
        with self.distance_lock:
            self.latest_distance = hit_data["Distance_m"]
            self.latest_confidence = hit_data["Correlation_Score"]

            current_time = time.time()

            if self.initial_distance_time is None:
                self.initial_distance_time = current_time

            self.latest_distance_time = (
                current_time - self.initial_distance_time
            )

    def get_live_iq(self):
        """
        Return a safe copy of the newest stored IQ samples.
        """
        with self.iq_lock:
            if self.latest_iq is None:
                return None, None

            return (
                self.latest_iq.copy(),
                self.latest_iq_timestamp
            )

    def get_distance(self):
        with self.distance_lock:

            if self.latest_distance is None:
                return None

            return {
                "time": self.latest_distance_time,
                "distance": self.latest_distance,
                "confidence": self.latest_confidence
            }


    def compute_fft(self, chunk):
        """
        Compute the windowed FFT magnitude in decibels.
        """
        chunk = np.asarray(chunk, dtype=np.complex64)

        window = np.hanning(len(chunk))

        fft_result = np.fft.fftshift(
            np.fft.fft(chunk * window)
        )

        power_db = 20 * np.log10(
            np.abs(fft_result) + 1e-12
        )

        return power_db.tolist()

    def get_fft_frame(self):
        """
        Create an FFT frame from the latest RX samples
        and the transmitted reference chirp.
        """

        iq_data, timestamp = self.get_live_iq()

        if iq_data is None:
            return None

        if len(iq_data) < self.fft_size:
            return None

        # RX FFT
        window = np.hanning(len(iq_data))
        fft_result = np.fft.fftshift(np.fft.fft(iq_data * window))
        power_db = 20 * np.log10(np.abs(fft_result) + 1e-12)

        # Downscaling array to fit the 2048-bin GUI
        step = len(power_db) // self.fft_size
        rx_fft = power_db[::step][:self.fft_size].tolist()

        # TX FFT
        tx_fft = None

        if self.tx_reference is not None:
            tx_active = self.tx_reference[np.abs(self.tx_reference) > 0.1]
            if len(tx_active) > 0:
                window_tx = np.hanning(len(tx_active))
                fft_results_tx = np.fft.fftshift(np.fft.fft(tx_active * window_tx))
                power_db_tx = 20 * np.log10(np.abs(fft_results_tx) + 1e-12)
                step_tx = len(power_db_tx) // self.fft_size
                tx_fft = power_db_tx[::step_tx][:self.fft_size].tolist()

        return {
            "timestamp": timestamp,
            "fft": rx_fft,
            "tx_fft": tx_fft,
            "fft_size": self.fft_size,
            "sample_rate": self.sample_rate
        }

    def _register_routes(self):
        """
        Register Flask API routes.
        """

        @self.app.route("/")
        def home():
            return """
            <h1>Flask SDR Server Running</h1>
            <p>Status: OK</p>
            <p>Endpoints:</p>
            <ul>
                <li>/status - server status</li>
                <li>/spectrum - latest live FFT frame</li>
                <li>/waterfall - latest live FFT frame</li>
            </ul>
            """

        @self.app.route("/status")
        def status():
            iq_data, timestamp = self.get_live_iq()

            return jsonify({
                "status": "running",
                "live_data_available": iq_data is not None,
                "last_update": timestamp,
                "sample_rate": self.sample_rate,
                "fft_size": self.fft_size
            })

        @self.app.route("/spectrum")
        def spectrum():
            frame = self.get_fft_frame()

            if frame is None:
                return jsonify({
                    "error": "Live SDR data is not available yet"
                }), 503

            return jsonify(frame)

        @self.app.route("/waterfall")
        def waterfall():
            frame = self.get_fft_frame()

            if frame is None:
                return jsonify({
                    "error": "Live SDR data is not available yet"
                }), 503

            return jsonify(frame)

        @self.app.route("/distance_by_time")
        def distance_by_time():
            frame = self.get_distance()

            if frame is None:
                return jsonify({
                    "error": "Live distance data is not available yet"
                }), 503

            return jsonify(frame)

    def run_server(self, dsp_queue=None, iq_queue=None, stop_event=None):
        """
        Start the Flask server.
        """

        if dsp_queue is not None and stop_event is not None:
            threading.Thread(
                target=self.consume_dsp_queue,
                args=(dsp_queue, stop_event),
                daemon=True
            ).start()

        if iq_queue is not None and stop_event is not None:
            threading.Thread(
                target=self.consume_iq_queue,
                args=(iq_queue, stop_event),
                daemon=True
            ).start()

        print("Flask SDR server running at http://127.0.0.1:5001")

        self.app.run(
            host="0.0.0.0",
            port=5001,
            debug=False,
            use_reloader=False,
            threaded=True
        )


if __name__ == "__main__":
    server = FlaskSDRServer()
    server.run_server()


# -----------------------------
# STANDALONE EXECUTION
# -----------------------------
if __name__ == "__main__":
    run_server()
