# Constants file
from enum import Enum
import numpy as np
from qiskit.circuit import Parameter


class QUBIT_PARA(Enum):
    QUBIT = 0
    PI = np.pi
    THETA = Parameter('theta')
    PHI = Parameter('phi')
