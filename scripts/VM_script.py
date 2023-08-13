from src.quantumcircuit import QC
from src.vm_backend.QASM_backend import QASM_Simulator
import numpy as np


pi = np.pi
'''
Simple example of VM
'''
qc = QC.Qutrit_circuit(4, None)
qc.add_gate('x01', first_qutrit_set=1)
qc.add_gate('x12', first_qutrit_set=1)
qc.add_gate('rx12', first_qutrit_set=0, parameter=[pi / 2])
# qc.add_gate('CNOT', first_qutrit_set=3, second_qutrit_set=1)
qc.measure_all()


backend = QASM_Simulator(qc=qc)
backend.run(num_shots=2048)
print(backend.return_final_state())
print("-------------")
backend.plot(plot_type="histogram")