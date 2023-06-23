from src.quantumcircuit.QC import Qutrit_circuit
from src.vm_backend.QASM_backend import QASM_Simulator
from src.decomposition.transpilation import SU3_matrices
import numpy as np

"""
Test adding function
"""
# qc_1 = Qutrit_circuit(3, None)
# qc_1.add_gate("hdm", first_qutrit_set=0)
# qc_1.add_gate("rx01", first_qutrit_set=0, parameter=[np.pi])
#
# qc_2 = Qutrit_circuit(3, None)
# qc_2.add_gate("rx01", first_qutrit_set=0, parameter=[np.pi])
# qc_2.add_gate("hdm", first_qutrit_set=0)
# qc_2.measure_all()
#
# qc = qc_1 + qc_2
# qc.draw()

"""
Test decomposition function
"""
qc_1 = Qutrit_circuit(1, None)
qc_1.add_gate("hdm", first_qutrit_set=0)
tmp_matrix = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 0.0, 0.0]], dtype=complex)
decomposer = SU3_matrices(su3=tmp_matrix, qutrit_index=0, n_qutrits=1)
qc_sub = decomposer.decomposed_into_qc()
qc_sub.draw()
qc = qc_1 + qc_sub
qc.add_gate("hdm", first_qutrit_set=0)
qc.measure_all()
qc.draw()
"""
Simulation
"""
backend = QASM_Simulator(qc=qc)
backend.run(num_shots=2048)
print(backend.return_final_state())
backend.plot(plot_type="histogram")
