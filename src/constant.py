# Constants file
from enum import Enum


class QubitParameters(Enum):
    """
    Qutrit related Parameters
    """
    # Frequency unit
    MHZ = 1.0e6
    GHZ = 1.0e9

    # th-Qubit used for running: used for nairobi
    QUBIT_CHANGE_TYPE1 = 6

    # Other constants
    CBIT = 0

    # Scale
    SCALE_FACTOR = 1.0e-14

