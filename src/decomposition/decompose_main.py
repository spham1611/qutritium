import numpy as np
import pickle
import matplotlib.pyplot as plt
from qiskit import *
from src.quantumcircuit import QC
from transpilation import Pulse_Wrapper
from src.pulse import Pulse01, Pulse12
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit_ibm_provider.ibm_provider import IBMProvider
pi = np.pi
'''
Simple example to test VM 
'''
provider = IBMProvider()
backend = provider.get_backend('ibm_nairobi')
drive_freq = backend.properties().qubit_property(0)['frequency'][0]
anha_freq = backend.properties().qubit_property(0)['anharmonicity'][0]
qc_list = []

pulse01 = Pulse01(frequency=drive_freq, x_amp=0.13884080278518465, duration=160)
pulse12 = Pulse12(frequency=drive_freq + anha_freq, x_amp=0.13884080278518465, duration=160, pulse01=pulse01)
n_qutrit = 1

"""
Example to reconstruct state 0, 1 and 2
"""
# for i in range(3):
#     qc = QC.Qutrit_circuit(n_qutrit, None)
#     if i == 0:
#         qiskit_circuit = QuantumCircuit(n_qutrit, n_qutrit)
#         qiskit_circuit.measure(range(n_qutrit), range(n_qutrit))
#         qc_list.append(qiskit_circuit)
#     else:
#         if i == 1:
#             qc.add_gate('rx01', first_qutrit_set=0, parameter=[pi])
#         elif i == 2:
#             qc.add_gate('rx01', first_qutrit_set=0, parameter=[pi])
#             qc.add_gate('rx12', first_qutrit_set=0, parameter=[pi])
#
#         pulse_wrap = Pulse_Wrapper(pulse01, pulse12, qc=qc, native_gates=['x12', 'x01',
#                                                                           'rx12', 'rx01',
#                                                                           'rz01', 'rz12',
#                                                                           'z01', 'z12'])
#         pulse_wrap.decompose()
#         pulse_wrap.convert_to_pulse_model()
#         pulse_wrap.print_decompose_pulse()
#         sched = pulse_wrap.pulse_model_to_qiskit()
#         print(i)
#         print(sched)
#
#         qiskit_circuit = QuantumCircuit(n_qutrit, n_qutrit)
#         custom_gate = Gate(name='Decomposed_Schedule', num_qubits=n_qutrit, params=[1])
#         qiskit_circuit.append(custom_gate, range(n_qutrit))
#         qiskit_circuit.measure(range(n_qutrit), range(n_qutrit))
#         qiskit_circuit.add_calibration(custom_gate, range(n_qutrit), sched)
#         qc_list.append(qiskit_circuit)
#
# print(qc_list)
# job = execute(qc_list, backend, shots=8192, meas_level=1, meas_return="single")
# print(job.status())
# print(job)

"""
Experiment to test decomposition with U_d centric
"""
qc = QC.Qutrit_circuit(1, None)
qc.add_gate("hdm", first_qutrit_set=0)
pulse_wrap = Pulse_Wrapper(pulse01, pulse12, qc=qc, native_gates=['x12', 'x01',
                                                                  'rx12', 'rx01',
                                                                  'rz01', 'rz12',
                                                                  'z01', 'z12'], backend=backend)
pulse_wrap.decompose()
pulse_wrap.print_decompose_ins()
pulse_wrap.convert_to_pulse_model()
pulse_wrap.print_decompose_pulse()
pulse_wrap.pulse_model_to_qiskit()
pulse_wrap.print_qiskit_sched()
