from src.quantumcircuit import QC
from src.pulse import Pulse01, Pulse12
from src.decomposition.transpilation import Pulse_Wrapper
from qiskit_ibm_provider.ibm_provider import IBMProvider
import numpy as np

pi = np.pi
'''
Simple example of decomposition
'''

"""
Define calibrated pulses and device backend
"""
provider = IBMProvider()
backend = provider.get_backend('ibm_nairobi')
drive_freq = backend.properties().qubit_property(0)['frequency'][0]
anha_freq = backend.properties().qubit_property(0)['anharmonicity'][0]
qc_list = []

pulse01 = Pulse01(frequency=drive_freq, x_amp=0.13884080278518465, duration=160)
pulse12 = Pulse12(frequency=drive_freq + anha_freq, x_amp=0.13884080278518465, duration=160, pulse01=pulse01)

"""
Define qutrit circuit
"""
qc = QC.Qutrit_circuit(2, None)
qc.add_gate('hdm', first_qutrit_set=0)
qc.add_gate('rx01', first_qutrit_set=0, parameter=[pi / 2])
qc.add_gate('x01', first_qutrit_set=1)

"""
Decomposition
"""
pulse_wrap = Pulse_Wrapper(pulse01, pulse12, qc=qc, native_gates=['x12', 'x01',
                                                                  'rx12', 'rx01',
                                                                  'rz01', 'rz12',
                                                                  'z01', 'z12'], backend=backend)
pulse_wrap.decompose()
pulse_wrap.print_decompose_ins()
pulse_wrap.convert_to_pulse_model()
pulse_wrap.print_decompose_pulse()
pulse_wrap.pulse_model_to_qiskit()
pulse_wrap.print_qiskit_schedule()
