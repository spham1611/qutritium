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


class GateSchedule:
    """ Provides custom schedules
    Here is list of available static methods of "GateSchedule" class:
        * freq_gaussian(): sweep over frequency
        * x_amp_gaussian(): sweep over x_amp
    """

    @staticmethod
    def freq_gaussian(
            backend: IBMBackend,
            frequency: float,
            pulse_model: Pulse,
            qubit: int,
    ) -> ScheduleBlock:
        """

        Args:
            backend:
            frequency:
            pulse_model:
            qubit:

        Returns:
            ScheduleBlock: object
        """
        with pulse.build(backend=backend) as gaussian_schedule:
            drive_chan = pulse.channels.DriveChannel(qubit)
            pulse.set_frequency(frequency, drive_chan)
            pulse.play(pulse.Gaussian(duration=pulse_model.duration,
                                      sigma=pulse_model.sigma, amp=pulse_model.x_amp), drive_chan)
        return gaussian_schedule

    @staticmethod
    def x_amp_gaussian(
            backend: IBMBackend,
            x_amp: float,
            pulse_model: Pulse,
            qubit: int,
    ) -> ScheduleBlock:
        """

        Args:
            backend:
            x_amp:
            pulse_model:
            qubit:

        Returns:

        """
        with pulse.build(backend=backend) as gaussian_schedule:
            drive_chan = pulse.channels.DriveChannel(qubit)
            pulse.set_frequency(pulse_model.frequency, drive_chan)
            pulse.play(pulse.Gaussian(duration=pulse_model.duration,
                                      sigma=pulse_model.sigma, amp=x_amp), drive_chan)
        return gaussian_schedule

    @staticmethod
    def drag(
            backend: IBMBackend,
            beta: float,
            pulse_model: Pulse,
            qubit: int
    ) -> ScheduleBlock:
        """

        Args:
            backend:
            beta:
            pulse_model:
            qubit:

        Returns:

        """
        with pulse.build(backend=backend) as beta_sweep:
            drive_chan = pulse.channels.DriveChannel(qubit)
            pulse.set_frequency(pulse_model.frequency, drive_chan)
            pulse.play(pulse.Drag(duration=pulse_model.duration, sigma=pulse_model.sigma, amp=pulse_model.x_amp,
                                  beta=beta), drive_chan)
        return beta_sweep

    @staticmethod
    def delay(
            backend: IBMBackend,
            qubit: int,
            delay_time: int = 22496
    ) -> ScheduleBlock:
        """

        Args:
            backend:
            qubit:
            delay_time:

        Returns:

        """
        with pulse.build(backend=backend) as delay_schedule:
            drive_chan = pulse.channels.DriveChannel(qubit)
            pulse.delay(delay_time, drive_chan)
        return delay_schedule


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
            pulse.shift_phase(phase=self.value * coeff, channel=pulse.channels.DriveChannel(self.channel))
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
            pulse.set_frequency(frequency=self.value, channel=pulse.channels.DriveChannel(self.channel))
        return schedule
