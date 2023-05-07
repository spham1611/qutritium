from src.quantumcircuit import QC
import numpy as np
from transpilation import Pulse_Wrapper
from src.pulse import Pulse01, Pulse12

pi = np.pi
'''
Simple example to test VM 
'''
pulse01 = Pulse01(frequency=6 * 10e6, x_amp=1.0)
pulse12 = Pulse12(frequency=5 * 10e6, x_amp=0.5, pulse01=pulse01)
qc = QC.Qutrit_circuit(1, None)
qc.add_gate('rx12', first_qutrit_set=0, parameter=[pi])
qc.add_gate('rx01', first_qutrit_set=0, parameter=[pi])
pulse_wrap = Pulse_Wrapper(pulse01, pulse12, qc=qc, native_gates=['x12', 'x01',
                                                                  'rx12', 'rx01',
                                                                  'rz01', 'rz12',
                                                                  'z01', 'z12'])
pulse_wrap.decompose()
# pulse_wrap.print_decompose_ins()
pulse_wrap.convert_to_pulse_model()
pulse_wrap.print_decompose_pulse()
sched = pulse_wrap.pulse_model_to_qiskit()
print(sched)