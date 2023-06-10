# MIT License
#
# Copyright (c) [2023] [son pham, tien nguyen, bach bao]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Calibration utility functions for gate operations and other pulse type operations"""
from qiskit_ibm_provider import IBMBackend
from qiskit.pulse.schedule import ScheduleBlock
from qiskit import pulse

from src.pulse import Pulse


class Pulse_Schedule:
    """
    Overwriting the phase offset and gaussian schedule method for pulse model
    """

    @staticmethod
    def single_pulse_schedule(backend: IBMBackend,
                              pulse_model: Pulse,
                              name: str = '',
                              drive_phase: float = 0,
                              drive_beta: float = 0,
                              channel: int = 0) -> ScheduleBlock:
        """

        Args:
            backend:
            pulse_model:
            name:
            drive_phase:
            drive_beta:
            channel:

        Returns:

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
    def single_pulse_gaussian_schedule(backend: IBMBackend,
                                       pulse_model: Pulse,
                                       channel: int = 0,
                                       name: str = '') -> ScheduleBlock:
        """

        Args:
            backend:
            pulse_model:
            channel:
            name:

        Returns:

        """
        with pulse.build(backend=backend) as drive_schedule:
            drive_chan = pulse.drive_channel(channel)
            pulse.set_frequency(pulse_model.frequency, drive_chan)
            pulse.play(pulse.Gaussian(duration=pulse_model.duration, amp=pulse_model.x_amp,
                                      sigma=pulse_model.duration / 4, name=name), drive_chan)
        return drive_schedule


class Shift:
    """

    """

    def __init__(self, shift_type: str, value: float, channel: int) -> None:
        """

        Args:
            shift_type:
            value:
            channel:
        """
        self.type = shift_type


class Shift_phase:
    """

    """

    def __init__(self, value: float, channel: int,
                 backend: IBMBackend, subspace: str = "01") -> None:
        """

        :param value:
        :param channel:
        :param subspace:
        """
        self.value = value
        self.channel = channel
        self.subspace = subspace
        self.backend = backend

    def generate_qiskit_phase(self, coeff: int = 1) -> ScheduleBlock:
        """
        :param coeff:
        :return:
        """
        with pulse.build(backend=self.backend) as schedule:
            pulse.shift_phase(phase=self.value * coeff, channel=pulse.drive_channel(self.channel))
        return schedule

    def generate_qiskit_phase_offset(self, gate_pulse: ScheduleBlock) -> ScheduleBlock:
        """

        :param gate_pulse:
        :return:
        """
        if self.subspace == "01":
            pos_schedule = self.generate_qiskit_phase(coeff=1)
            neg_schedule = self.generate_qiskit_phase(coeff=-1)
        elif self.subspace == "12":
            pos_schedule = self.generate_qiskit_phase(coeff=-1)
            neg_schedule = self.generate_qiskit_phase(coeff=1)
        else:
            raise Exception("The shift phase is not in the defined subspace")
        schedule = pos_schedule + gate_pulse
        schedule += neg_schedule
        return schedule


class Set_frequency:
    """

    """

    def __init__(self, value: float,
                 backend: IBMBackend, channel: int) -> None:
        """

        :param value:
        :param channel:
        """
        self.value = value
        self.channel = channel
        self.backend = backend

    def generate_qiskit_freq(self) -> ScheduleBlock:
        """

        :return:
        """
        with pulse.build(backend=self.backend) as schedule:
            pulse.set_frequency(frequency=self.value, channel=pulse.drive_channel(self.channel))
        return schedule
