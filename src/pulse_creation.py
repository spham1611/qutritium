"""Calibration utility functions for gate operations and other"""
from qiskit.pulse.schedule import ScheduleBlock
from qiskit import pulse
from qiskit.circuit import Parameter
from src.calibration import backend
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
                             name: str = '$X^{01}$',
                             channel: int = 0) -> ScheduleBlock:
        """

        :param channel:
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
            drive_chan = pulse.drive_channel(channel)
            pulse.set_frequency(drive_freq, drive_chan)
            with pulse.phase_offset(drive_phase):
                pulse.play(pulse.Drag(drive_duration, drive_amp, drive_sigma, drive_beta), drive_chan)

        return drive_schedule

    @staticmethod
    def single_gate_schedule_gaussian(drive_freq: float,
                                      drive_duration: int,
                                      drive_amp: float,
                                      name: str = '$X^{01}$',
                                      channel: int = 0) -> ScheduleBlock:
        """

        :param channel:
        :param drive_freq:
        :param drive_duration:
        :param drive_amp:
        :param name:
        :return:
        """
        if name != '$X^{01}$' and name != '$X^{12}$':
            raise ValueError
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(channel)
            pulse.set_frequency(drive_freq, drive_chan)
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
                              channel: int = 0) -> ScheduleBlock:
        """

        :return:
        """
        drive_sigma = pulse_model.duration / 4
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(channel)
            pulse.set_frequency(pulse_model.frequency, drive_chan)
            with pulse.phase_offset(drive_phase):
                pulse.play(pulse.Drag(pulse_model.duration, pulse_model.x_amp,
                                      drive_sigma, drive_beta, name=name), drive_chan)

        return drive_schedule

    @staticmethod
    def single_pulse_gaussian_schedule(pulse_model: Pulse, channel: int = 0, name: str = '') -> ScheduleBlock:
        """

        :param channel:
        :param name:
        :param pulse_model:
        :return:
        """
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(channel)
            pulse.set_frequency(pulse_model.frequency, drive_chan)
            pulse.play(pulse.Gaussian(duration=pulse_model.duration, amp=pulse_model.x_amp,
                                      sigma=pulse_model.duration / 4, name=name), drive_chan)
        return drive_schedule


class Shift:
    def __init__(self, shift_type: str, value: float, channel: int) -> None:
        self.type = shift_type
        self.value = value
        self.channel = pulse.drive_channel(channel)

    def generate_qiskit_phase(self, coeff: int = 1) -> ScheduleBlock:
        with pulse.build(backend=backend) as schedule:
            pulse.shift_phase(phase=self.value * coeff, channel=self.channel)
        return schedule

    def generate_qiskit_freq(self) -> ScheduleBlock:
        with pulse.build(backend=backend) as schedule:
            pulse.shift_frequency(frequency=self.value, channel=self.channel)
        return schedule

    def create_qiskit_effect(self, gate_pulse: ScheduleBlock) -> ScheduleBlock:
        if self.type == "phase_offset":
            pos_schedule = self.generate_qiskit_phase(coeff=1)
            neg_schedule = self.generate_qiskit_phase(coeff=-1)
            schedule = pos_schedule + gate_pulse
            schedule += neg_schedule
            return schedule
        elif self.type == "shift_phase":
            pos_schedule = self.generate_qiskit_phase(coeff=1)
            schedule = pos_schedule + gate_pulse
            return schedule
        elif self.type == "shift_freq":
            pos_schedule = self.generate_qiskit_freq()
            schedule = pos_schedule + gate_pulse
            return schedule
