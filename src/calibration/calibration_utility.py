"""Calibration Helpful functions"""
from src.calibration import backend
from src.constant import QUBIT_PARA
from src.calibration import QUBIT_VAL, ANHAR
import qiskit.pulse as pulse


def g01(amp, phi, **kwargs) -> pulse:
    """

    :param amp:
    :param phi:
    :param kwargs: contains: pulse duration, beta and drive_freq of the backend
    :return:
    """
    if len(kwargs) != 3:
        raise Exception

    with pulse.build(backend=backend) as g01_pulse:
        pulse.set_frequency(kwargs["drive_freq"], pulse.drive_channel(QUBIT_PARA.QUBIT.value))
        with pulse.phase_offset(phi, pulse.drive_channel(QUBIT_PARA.QUBIT.value)):
            pulse.play(pulse.Drag(kwargs['dur'], amp, kwargs['dur'] / 4, kwargs['beta'],
                                  '$\mathcal{G}^{01}$' + f'({round(amp, 2)}, {round(phi, 2)})'),
                       pulse.drive_channel(QUBIT_PARA.QUBIT.value))
    return g01_pulse


def g12(amp, phi, **kwargs) -> pulse:
    """

    :param amp:
    :param phi:
    :param kwargs: contains: pulse duration, beta and drive_freq of the backend
    :return:
    """
    if len(kwargs) != 3:
        raise Exception

    with pulse.build(backend=backend) as g12_pulse:
        pulse.set_frequency(kwargs["drive_freq"] + ANHAR, pulse.drive_channel(QUBIT_PARA.QUBIT.value))
        with pulse.phase_offset(phi, pulse.drive_channel(QUBIT_VAL)):
            pulse.play(pulse.Drag(kwargs['dur'], amp, kwargs['dur'] / 4, kwargs['beta'],
                                  r'$\mathcal{G}^{12}$' + f'({round(amp, 2)})'),
                       pulse.drive_channel(QUBIT_VAL))
    return g12_pulse
