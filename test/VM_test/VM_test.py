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
omega = np.exp(1j * 2 * np.pi / 3)

qc_1 = Qutrit_circuit(1, None)
qc_1.add_gate(gate_type="x01", first_qutrit_set=0)
qc_1.draw()
print("-------------")
U_ft = 1 / np.sqrt(3) * np.array([[omega, 1.0, np.conj(omega)],
                                  [1.0, 1.0, 1.0],
                                  [np.conj(omega), 1.0, omega]], dtype=complex)
decomposer_1 = SU3_matrices(su3=U_ft, qutrit_index=0, n_qutrits=1)
qc_sub_1 = decomposer_1.decomposed_into_qc()
qc_sub_1.draw()
print("-------------")

U_ft_dagger = np.matrix(U_ft).getH()
decomposer_2 = SU3_matrices(su3=U_ft_dagger, qutrit_index=0, n_qutrits=1)
qc_sub_2 = decomposer_2.decomposed_into_qc()
qc_sub_2.draw()
print("-------------")

qc = qc_sub_2 + qc_1 + qc_sub_1
qc.measure_all()
qc.draw()
print("-------------")

"""
Test to_all function
"""
# qc_1 = Qutrit_circuit(5, None)
# qc_1.add_gate("hdm", first_qutrit_set=0, to_all=True)
# qc_1.draw()

"""
Simulation
"""
backend = QASM_Simulator(qc=qc)
backend.run(num_shots=2048)
print(backend.return_final_state())
print("-------------")
backend.plot(plot_type="histogram")
