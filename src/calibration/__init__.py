"""Import packages and set up IBM vm_backend for calibration"""
from src.backend_ibm import BackEndDict
# from src.constant import QUBIT_PARA
# import numpy as np


backend, QUBIT_VAL = BackEndDict().default_backend()
# Constant values coming from the IBM quantum computer. Because those depend
# on computer, we will not save them in the constant.py
ANHAR = backend.qubit_properties(QUBIT_VAL).__getattribute__('anharmonicity')
DEFAULT_F01 = backend.qubit_properties(QUBIT_VAL).frequency
DEFAULT_F12 = DEFAULT_F01 + ANHAR
# DRIVE_FREQ = vm_backend.configuration().hamiltonian['vars'][f'wq{QUBIT_VAL}'] / (2 * np.pi)
