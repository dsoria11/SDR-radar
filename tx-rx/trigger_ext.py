""" 
trigger_ext.py
~~~~~~~~~~~~~~

Will be used to synchronize the transmit and receive operations of the SDR-enabled UHF Radar System.
Adds trigger support that isn't yet wrapped in the stock bladerf._bladerf bindings.
"""

from bladerf._bladerf import ffi, libbladeRF, _check_error

class TriggerRole:
    """ Enum for trigger roles. """
    DISABLED = 0
    MASTER = 1
    SLAVE = 2

class TriggerSignal:
    """ Enum for trigger signals. """
    J71_4 = 0
    J51_1 = 1
    MINI_EXP_1 = 2

def trigger_init(sdr, channel, signal=TriggerSignal.MINI_EXP_1):
    """ Channel: e.g. _bladerf.CHANNEL_TX(0) or _bladerf.CHANNEL_RX(0) """
    trig = ffi.new("struct bladerf_trigger *")
    ret = libbladeRF.bladerf_trigger_init(sdr.dev[0], channel, signal, trig)
    _check_error(ret)
    return trig

def trigger_arm(sdr, trig, role, arm=True):
    """ Arm or disarm the trigger. """
    trig.role = role 
    ret = libbladeRF.bladerf_trigger_arm(sdr.dev[0], trig, arm, 0, 0)
    _check_error(ret)

def trigger_fire(sdr, trig):
    """ Fire the trigger. """
    ret = libbladeRF.bladerf_trigger_fire(sdr.dev[0], trig)
    _check_error(ret)


