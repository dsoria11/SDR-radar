"""
gimbal_controller.py
~~~~~~~~~~~~~~~~~~~~
A serial wrapper to interface with the Arduino running the gimbal motors.
"""

import serial
import time

class GimbalController:
    def __init__(self, port='/dev/ttyXXXX', baudrate=9600):
        """ Initializes the serial connection to the Arduino. """
        self.port = port
        self.baudrate = baudrate
        self.ser = None

        try:
            # Connect to Arduino
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset upon connection
            print(f"[Gimbal] Successfully connected to Arduino on {self.port}")
        except Exception as e:
            print(f"[Gimbal Error] Could not connect to Arduino: {e}")

    def set_position(self, azimuth, elevation):
        # TODO: azimuth, elevation has to be in float16 (py.gpio) """ 
        """ Sends the euler angles to command the Arduino to move at the designated position. """
        self.azimuth = azimuth
        self.elevation = elevation

        if self.ser:
            # Format: "azimuth,elevation\n"
            command = f"{azimuth},{elevation}\n"
            self.ser.write(command.encode('utf-8'))

        print(f"[Gimbal] Position Updated -> Azimuth: {self.azimuth}°, Elevation: {self.elevation}°")