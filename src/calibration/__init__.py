"""Import packages and set up IBM backend for calibration"""
from src.backend_ibm import BackEndList
from src.constant import QUBIT_PARA
import numpy as np


backend, QUBIT_VAL = BackEndList().default_backend()
# Constant values coming from the IBM quantum computer. Because those depend
# on computer, we will not save them in the constant.py
ANHAR = backend.properties().qubits[QUBIT_VAL][3].value * QUBIT_PARA.GHZ.value
DEFAULT_F01 = backend.defaults().qubit_freq_est[QUBIT_VAL]
DEFAULT_F12 = DEFAULT_F01 + ANHAR
DRIVE_FREQ = backend.configuration().hamiltonian['vars'][f'wq{QUBIT_VAL}'] / (2 * np.pi)
