# Constants file
from enum import Enum


class QubitParameters(Enum):
    """
    Qutrit related Parameters
    """
    # Frequency unit
    MHZ = 1.0e6
    GHZ = 1.0e9

    # Time constants
    us = 1.0e-6
    ns = 1.0e-9

    # Other constants
    CBIT = 0

    # Scale
    SCALE_FACTOR = 1.0e-14

