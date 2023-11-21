from src.quantumcircuit import QC
from src.vm_backend.QASM_backend import QASM_Simulator
import numpy as np


pi = np.pi
'''
Simple example of VM
'''
qc = QC.Qutrit_circuit(2, None)
qc.add_gate("hdm", first_qutrit_set=0)
qc.add_gate("CNOT", first_qutrit_set=1,
            second_qutrit_set=0)
qc.measure_all()


backend = QASM_Simulator(qc=qc)
backend.run(num_shots=10000)
print(backend.return_final_state())
print("-------------")
backend.plot(plot_type="histogram")