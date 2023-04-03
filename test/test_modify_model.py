"""Internal use only not a real test!"""
# from src.calibration.transmission_reflection import TR_01, TR_12
# from src.calibration.rough_rabi import Rough_Rabi01, Rough_Rabi12
from src.calibration.drag_dephase import DragDP01, DragDP12
from src.calibration.drag_leakage import DragLK01, DragLK12
from src.pulse import Pulse01, Pulse12


# Test DRAGDP and DRAGLK
pulse01 = Pulse01(
    frequency=4900000000.0,
    duration=144,
    x_amp=0.27789268017955304,
)
pulse12 = Pulse12(
    frequency=4800000000,
    x_amp=0.28655972253635953,
    duration=144,
    pulse01=pulse01
)
dp_01 = DragDP01(pulse_model=pulse01)
dp_01.run()
print(pulse01)
