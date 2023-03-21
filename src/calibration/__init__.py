"""Import packages and set up IBM backend for calibration"""
from qiskit import IBMQ
from src.constant import QUBIT_PARA
import numpy as np

# IBM Config
if not IBMQ.active_account():
    IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibm_nairobi')

# Constants value coming from the IBM quantum computer. Because those depend
# on computer, we will not save in the constant file
QUBIT_VAL = int(QUBIT_PARA.QUBIT.value)
ANHAR = backend.properties().qubits[QUBIT_VAL][3].value * QUBIT_PARA.GHZ.value
DEFAULT_F01 = backend.defaults().qubit_freq_est[QUBIT_VAL]
DEFAULT_F12 = DEFAULT_F01 + ANHAR
DRIVE_FREQ = backend.configuration().hamiltonian['vars'][f'wq{QUBIT_VAL}'] / (2 * np.pi)
