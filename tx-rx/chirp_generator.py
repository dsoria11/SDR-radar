"""
chirp_generator.py
~~~~~~~~~~~~~~~~~~~

Generates the unpadded complex reference chirp and builds the sample-aligned
Tx frame (N_off zeros -> N_c chirp samples -> N_post zeros) for SC16_Q11.
"""

import numpy as np

def make_tx_frame(chirp, n_rx, n_off, amplitude):
    """
    Zeros, then the chirp at a known sample offset, then zeros. Length = capture length,
    so Tx and Rx started on the same trigger line up sample-for-sample.
    """
    frame = np.zeros(n_rx, dtype=np.complex64)
    frame[n_off : n_off + chirp.size] = amplitude * chirp
    return frame

def gen_radar_chirp(sample_rate, chirp_bandwidth, pulse_duration, n_rx=65536, n_off=8192, amplitude=0.5):
    """
    Returns:
        - tx_bytes: SC16_Q11 interleaved I/Q bytes of length n_rx * 4
        - chirp: unpadded unit-amplitude complex64 reference chirp of length N_c (40,000)
        - tx_frame: full complex64 frame of length n_rx (65,536) for GUI/reference
    """
    num_chirp_samples = int(round(sample_rate * pulse_duration))
    t = np.linspace(0.0, pulse_duration, num_chirp_samples, endpoint=False)

    f0 = -chirp_bandwidth / 2.0
    slope = chirp_bandwidth / pulse_duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * slope * t**2)

    # Unit-amplitude active chirp (N_c samples)
    chirp = np.exp(1j * phase).astype(np.complex64)

    # Full TX frame with N_off leading zeros and N_post trailing zeros
    tx_frame = make_tx_frame(chirp, n_rx, n_off, amplitude)

    i_samples = np.clip(np.round(np.real(tx_frame) * 2048.0), -2048, 2047).astype(np.int16)
    q_samples = np.clip(np.round(np.imag(tx_frame) * 2048.0), -2048, 2047).astype(np.int16)

    iq_interleaved = np.empty(2 * n_rx, dtype=np.int16)
    iq_interleaved[0::2] = i_samples
    iq_interleaved[1::2] = q_samples

    return iq_interleaved.tobytes(), chirp, tx_frame

def ca_cfar_detect(correlation, guard_cells=20, ref_cells=200, pfa_factor=6.0):
    """
    Implements a Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection algorithm.
    
    Args:
        correlation (np.ndarray): The correlation output from the matched filter.
        guard_cells (int): Number of guard cells around the cell under test (CUT).
        ref_cells (int): Number of reference cells used to estimate the noise floor.
        pfa_factor (float): The desired probability of false alarm factor.
        
    Returns:
        peak_index
        peak_value
        threshold
        peak_value > threshold
    """
    peak_index = np.argmax(correlation)
    peak_value = correlation[peak_index]

    # Define the start and end indices for the reference cells
    lo = max(0, peak_index - guard_cells - ref_cells)
    hi = min(len(correlation), peak_index + guard_cells + ref_cells)
    guard_lo = max(0, peak_index - guard_cells)
    guard_hi = min(len(correlation), peak_index + guard_cells)

    # Reference window (everything in [lo, hi) excluding the guarded region around the peak)
    ref_window = np.concatenate([
        correlation[lo:guard_lo], 
        correlation[guard_hi:hi]
    ])

    noise_est = np.mean(ref_window)  
    threshold = noise_est * pfa_factor  # Set threshold based on noise estimation
    snr = 20 * np.log10(peak_value / noise_est) if noise_est != 0 else 0  # signal-to-noise ratio in dB

    return peak_index, peak_value, threshold, peak_value > threshold, snr

def ca_cfar_detect_gated(correlation, max_lag_samples, guard_cells=9, ref_cells=32, pfa_factor=12.0):
    """
    Same as ca_cfar_detect, but restricts peak search to the first `max_lag_samples`
    (a physically plausible range gate), while still pulling reference cells from
    the full correlation array for a stable noise-floor estimate.
    """
    gated = correlation[:max_lag_samples]
    peak_index = np.argmax(gated)
    peak_value = gated[peak_index]

    lo = max(0, peak_index - guard_cells - ref_cells)
    hi = min(len(correlation), peak_index + guard_cells + ref_cells)
    guard_lo = max(0, peak_index - guard_cells)
    guard_hi = min(len(correlation), peak_index + guard_cells)

    ref_window = np.concatenate([correlation[lo:guard_lo], correlation[guard_hi:hi]])
    noise_est = np.mean(ref_window)
    threshold = noise_est * pfa_factor

    return peak_index, peak_value, threshold, peak_value > threshold