"""Calibration utility functions for gate operation"""
from qiskit.pulse.schedule import ScheduleBlock
from src.calibration import backend, QUBIT_VAL
from qiskit import pulse


class Gate_Schedule:
    """Static class"""

    @staticmethod
    def single_gate_schedule(drive_freq,
                             drive_phase,
                             drive_duration,
                             drive_amp,
                             drive_beta, /) -> ScheduleBlock:
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
