"""Calibration utility functions for gate operations and other"""
from qiskit.pulse.schedule import ScheduleBlock
from src.calibration import backend, QUBIT_VAL
from qiskit import pulse


class Gate_Schedule:
    """Static class"""

    @staticmethod
    def single_gate_schedule(drive_freq: float,
                             drive_duration: int,
                             drive_amp: float,
                             drive_phase: int = 0,
                             drive_beta: float = 0.0,
                             ) -> ScheduleBlock:
        """

        :param drive_freq:
        :param drive_phase:
        :param drive_duration:
        :param drive_amp:
        :param drive_beta:
        :return:
        """
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
                                      mode: int = 1,
                                      ) -> ScheduleBlock:
        """

        :param mode:
        :param drive_freq:
        :param drive_duration:
        :param drive_amp:
        :return:
        """
        name = '$X^{01}$' if mode == 1 else '$X^{12}$'
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(drive_freq, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Gaussian(duration=drive_duration, amp=drive_amp, sigma=drive_duration / 4,
                                      name=name), drive_chan)
        return drive_schedule
