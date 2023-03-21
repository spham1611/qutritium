# Constants file
from enum import Enum


class QUBIT_PARA(Enum):
    # Frequency unit
    KHZ = 1.0e3
    MHZ = 1.0e6
    GHZ = 1.0e9

    # Other constants
    CBIT = 0
    QUBIT = 6
    ACQUIRE_ALIGNMENT = 16
    PULSE_ALIGNMENT = 16
    GRANULARITY = 16
    LCM = 16

    # Scale
    SCALE_FACTOR = 1e-7
