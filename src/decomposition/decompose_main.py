from src.quantumcircuit import QC
import numpy as np
from transpilation import Pulse_Wrapper

pi = np.pi
'''
Simple example to test VM 
'''
qc = QC.Qutrit_circuit(1, None)
qc.add_gate('hdm', first_qutrit_set=0)
qc.add_gate('x12', first_qutrit_set=0)
pulse_wrap = Pulse_Wrapper(qc=qc, native_gates=['x12', 'x01',
                                                'rx12', 'rx01',
                                                'rz01', 'rz12',
                                                'z01', 'z12'])
pulse_wrap.decompose()
pulse_wrap.print_decompose_ins()
