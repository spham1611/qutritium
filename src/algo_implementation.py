# MIT License
#
# Copyright (c) [2023] [son pham, tien nguyen, bach bao]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from src.quantumcircuit.QC import Qutrit_circuit
from src.vm_backend.QASM_backend import QASM_Simulator
from src.decomposition.transpilation import Pulse_Wrapper, SU3_matrices
from src.pulse import Pulse01, Pulse12
from qiskit_ibm_provider.ibm_provider import IBMProvider

pi = np.pi

"""
Hardware configuration 
"""

provider = IBMProvider()
backend = provider.get_backend('ibm_nairobi')
drive_freq = backend.properties().qubit_property(0)['frequency'][0]
anha_freq = backend.properties().qubit_property(0)['anharmonicity'][0]
qc_list = []

pulse01 = Pulse01(frequency=drive_freq, x_amp=0.13884080278518465, duration=160)
pulse12 = Pulse12(frequency=drive_freq + anha_freq, x_amp=0.13884080278518465, duration=160, pulse01=pulse01)
n_qutrit = 1

"""
Algorithm implementation
"""
matrix_array = [np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=complex),
                np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]], dtype=complex),
                np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]], dtype=complex),
                np.array([[0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=complex),
                np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0]], dtype=complex),
                np.array([[0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0]], dtype=complex)]

"""
Theoretically implementation
"""
final_value = []
for i in range(6):
    qc = Qutrit_circuit(1, None)
    qc.add_gate("hdm", first_qutrit_set=0)
    decomposer = SU3_matrices(su3=matrix_array[i], qutrit_index=0, n_qutrits=1)
    qc_sub = decomposer.decomposed_into_qc()
    qc += qc_sub
    qc.add_gate("hdm", first_qutrit_set=0)
    qc.measure_all()
    backend = QASM_Simulator(qc=qc)
    backend.run(num_shots=2048)
    final_value.append(backend.return_final_state())
print(final_value)
"""
Hardware implementation
"""
# pulse_wrap = Pulse_Wrapper(pulse01, pulse12, qc=qc, native_gates=['x12', 'x01',
#                                                                   'rx12', 'rx01',
#                                                                   'rz01', 'rz12',
#                                                                   'z01', 'z12'], backend=backend)
# pulse_wrap.decompose()
# pulse_wrap.print_decompose_ins()
# pulse_wrap.convert_to_pulse_model()
# pulse_wrap.print_decompose_pulse()
# pulse_wrap.pulse_model_to_qiskit()
# pulse_wrap.print_qiskit_schedule()