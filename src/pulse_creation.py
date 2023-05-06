"""Calibration utility functions for gate operations and other"""
from qiskit.pulse.schedule import ScheduleBlock
from qiskit import pulse
from qiskit.circuit import Parameter
from src.calibration import backend, QUBIT_VAL
from typing import Union

from src.pulse import Pulse


class Gate_Schedule:
    """Static class"""

    @staticmethod
    def single_gate_schedule(drive_freq: Union[float, Parameter],
                             drive_amp: Union[float, Parameter],
                             drive_duration: int,
                             drive_phase: float = .0,
                             drive_beta: float = .0,
                             name: str = '$X^{01}$'
                             ) -> ScheduleBlock:
        """

        :param drive_freq:
        :param drive_phase:
        :param drive_duration:
        :param drive_amp:
        :param drive_beta:
        :param name:
        :return:
        """
        if name != '$X^{01}$' and name != '$X^{12}$':
            raise ValueError
        drive_sigma = drive_duration / 4
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(drive_freq, drive_chan)
            with pulse.phase_offset(drive_phase):
                pulse.play(pulse.Drag(drive_duration, drive_amp, drive_sigma, drive_beta), drive_chan)

        return drive_schedule

    @staticmethod
    def single_gate_schedule_gaussian(drive_freq: float,
                                      drive_duration: int,
                                      drive_amp: float,
                                      name: str = '$X^{01}$',
                                      ) -> ScheduleBlock:
        """

        :param drive_freq:
        :param drive_duration:
        :param drive_amp:
        :param name:
        :return:
        """
        if name != '$X^{01}$' and name != '$X^{12}$':
            raise ValueError
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(drive_freq, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Gaussian(duration=drive_duration, amp=drive_amp, sigma=drive_duration / 4,
                                      name=name), drive_chan)
        return drive_schedule


class Pulse_Schedule(Gate_Schedule):
    """
    Overwriting the phase offset and gaussian method for pulse model
    """
    @staticmethod
    def single_pulse_schedule(pulse_model: Pulse,
                              name: str = '',
                              drive_phase: float = 0,
                              drive_beta: float = 0,
                              ) -> ScheduleBlock:
        """

        :return:
        """
        drive_sigma = pulse_model.duration / 4
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(pulse_model.frequency, drive_chan)
            with pulse.phase_offset(drive_phase):
                pulse.play(pulse.Drag(pulse_model.duration, pulse_model.x_amp,
                                      drive_sigma, drive_beta, name=name), drive_chan)

        return drive_schedule

    @staticmethod
    def single_pulse_gaussian_schedule(pulse_model: Pulse, name: str = '') -> ScheduleBlock:
        """

        :param name:
        :param pulse_model:
        :return:
        """
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(pulse_model.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Gaussian(duration=pulse_model.duration, amp=pulse_model.x_amp,
                                      sigma=pulse_model.duration / 4, name=name), drive_chan)
        return drive_schedule

