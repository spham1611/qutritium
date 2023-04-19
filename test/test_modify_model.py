"""Internal use only not a real test!"""
from src.calibration.transmission_reflection import TR_01, TR_12
from src.calibration.rough_rabi import Rough_Rabi01, Rough_Rabi12
from src.calibration.drag_dephase import DragDP01, DragDP12
from src.calibration.drag_leakage import DragLK01, DragLK12
from src.pulse import Pulse01, Pulse12


# Test
pulse01 = Pulse01(
    duration=144,
    x_amp=0.2
)
pulse12 = Pulse12(
    duration=144,
    x_amp=0.2,
    pulse01=pulse01
)
tr_01 = TR_01(pulse_model=pulse01)
tr_01.set_up()
tr_01.modify_pulse_model(job_id='642d24e06db55d37f0428c4a')
print(pulse01)
rr_01 = Rough_Rabi01(pulse_model=pulse01)
rr_01.modify_pulse_model(job_id='642d943563ccbb964fbf4582')
print(pulse01)
tr_12 = TR_12(pulse_model=pulse12)
tr_12.set_up()
tr_12.modify_pulse_model(job_id='642e411d06c693164f2886ec')
print(pulse12)
rr_12 = Rough_Rabi12(pulse_model=pulse12)
rr_12.modify_pulse_model(job_id='642e719896284628a6e97e80')
print(pulse12)
dp_01 = DragDP01(pulse_model=pulse01)
dp_01.modify_pulse_model(job_id='642eaa686a82f3270f223db4')
print(pulse01)
dl_01 = DragLK01(pulse_model=pulse01)
dl_01.modify_pulse_model(job_id='642ebefd352042e321aaee29')
print(pulse01)
dp_12 = DragDP12(pulse_model=pulse12)
dp_12.modify_pulse_model(job_id='642fa9d6e30a5a9c380c9604')
print(pulse12)
dl_12 = DragLK12(pulse_model=pulse12)
dl_12.modify_pulse_model(job_id='6430f21b54badea5599123a1')
print(pulse12)
