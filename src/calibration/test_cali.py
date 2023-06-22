""""""
import warnings

from src.calibration.transmission_reflection import TR01, TR12
from src.calibration.rough_rabi import RoughRabi01, RoughRabi12
from src.calibration.discriminator import DiscriminatorQutrit
from src.pulse import Pulse01, Pulse12
from src.backend.backend_ibm import EffProvider

warnings.filterwarnings("ignore")

eff_provider = EffProvider(
    token='b751d05f9c3522f9d46a851e2830dfa0d6087643fcdda1588b781e2f349cbaa8c6d7b1dbec1ec3262857523289bb385ee8ad7b86a83c045ca69aba6b1bc4b3d8')
pulse01 = Pulse01(duration=144, x_amp=0.2)
pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)
tr_01 = TR01(pulse_model=pulse01, eff_provider=eff_provider, backend_name='ibmq_quito')
# tr_01.prepare_circuit()
# tr_01.run_monitor()
tr_01.modify_pulse_model('ci44lelejm3lf1cnm1dg')
print(pulse01)
rr_01 = RoughRabi01(pulse_model=pulse01, eff_provider=eff_provider, backend_name='ibmq_quito')
# rr_01.prepare_circuit()
# rr_01.run_monitor()
rr_01.modify_pulse_model('ci68glokbvp2ovrji920')
print(pulse01)
tr_12 = TR12(pulse_model=pulse12, eff_provider=eff_provider, backend_name='ibmq_quito')
# print(tr_12.default_frequency)
# tr_12.prepare_circuit()
# tr_12.run_monitor()
tr_12.modify_pulse_model('ci6j057hmv10icuudbrg')
print(pulse12)
rr_12 = RoughRabi12(pulse_model=pulse12, eff_provider=eff_provider, backend_name='ibmq_quito')
# rr_12.prepare_circuit()
# rr_12.run_monitor()
rr_12.modify_pulse_model('ci6kin68t9vighsij210')
print(pulse12)

# discriminator_circuit = DiscriminatorQutrit(eff_provider=eff_provider,
#                                             pulse_model=pulse12,
#                                             backend_name='ibmq_quito')
# discriminator_circuit.prepare_circuit()
# discriminator_circuit.run_monitor()



