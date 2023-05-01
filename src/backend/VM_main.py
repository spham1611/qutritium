from src.quantumcircuit import QC
import numpy as np
from src.backend import QASM_simulator

pi = np.pi
'''
Simple example to test VM 
'''
qc = QC.Qutrit_circuit(4, None)
qc.add_gate('x01', first_qutrit_set=1)
qc.add_gate('x12', first_qutrit_set=1)
qc.add_gate('rx12', first_qutrit_set=0, parameter=[pi / 2])
# qc.add_gate('CNOT', first_qutrit_set=3, second_qutrit_set=1)
qc.add_gate('measure', 0)
backend = QASM_simulator(QC=qc)
backend.run()
backend.result()
print(backend.get_counts())
backend.plot("histogram")
