"""Import packages and set up IBM backend for calibration"""
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
import qiskit.pulse as pulse
from ..constant import QUBIT_PARA


if not IBMQ.active_account():
    IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibm_oslo')
