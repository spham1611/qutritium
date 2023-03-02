"""Transmission and reflection techniques for 0-1 and 1-2
In here TR or tr stands for transmission and reflection
"""
from abc import ABC, abstractmethod
from numpy import linspace
from src.calibration import backend
from src.calibration import QUBIT_VAL, ANHAR
from qiskit.circuit import Parameter, Gate, QuantumCircuit
import qiskit.pulse as pulse
import numpy as np


class TR(ABC):
    """
    The class act as provider + regulator for transmission and reflection techniques
    """
    def __init__(self, duration=0) -> None:
        """
        Some parameters are restricted for design and read-only
        :param duration: in seconds
        """
        self._duration = duration
        self._freq = 0
        self._sweep_schedule = None
        self._freq_probe = None
        self._exp_circuits = None
        self.tr_create_pulse()
        self.tr_create_gate()
        self.run_monitor()

    @property
    def duration(self):
        return self._duration

    @property
    def freq(self):
        return self._freq

    @abstractmethod
    def tr_create_pulse(self):
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def tr_create_gate(self):
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def run_monitor(self):
        """

        :return:
        """
        raise NotImplementedError


class TR_01(TR):
    """"""
    def __init__(self, duration) -> None:
        """

        :param duration:
        """
        super().__init__(duration)

    @property
    def sweep_schedule(self):
        return self._sweep_schedule

    def tr_create_pulse(self):
        """

        :return:
        """
        est_freq = backend.configuration().hamiltonian['vars'][f'wq{QUBIT_VAL}']/(2*np.pi)
        max_freq, min_freq = est_freq + 30 * 1e6, est_freq - 30 * 1e6
        self._freq_probe = linspace(min_freq, max_freq, 100)
        self._freq = Parameter("freq")

        # Setting up pulse properties for 01
        with pulse.build(backend=backend) as sweep_schedule:
            drive_chan = pulse.drive_channel(qubit=QUBIT_VAL)
            pulse.delay(48, drive_chan)
            pulse.set_frequency(self._freq, drive_chan)
            pulse.play(pulse.Gaussian(
                duration=self.duration,   # input
                amp=0.2,
                sigma=self.duration / 4,
            ), drive_chan)
            self._sweep_schedule = sweep_schedule

        # If the system does not respond -> TimeoutError occurs
        if not self._sweep_schedule:
            raise TimeoutError

    def tr_create_gate(self):
        """

        :return:
        """
        g01 = Gate("g01", 1, self.freq)
        qc = QuantumCircuit(1, 1)
        qc.append(g01, [0])
        qc.measure(0, 0)
        qc.add_calibration(g01, (0, ), self._sweep_schedule, [self.freq])

        # Get the circuits from assigned frequencies
        self._exp_circuits = [qc.assign_parameters({self._freq: f}, inplace=False) for f in self._freq_probe]

    def run_monitor(self):
        pass


class TR_12(TR):
    """"""
    def __init__(self, duration) -> None:
        """

        :param duration:
        """
        self._tr_01 = TR_01(duration=duration)
        self._pulse01_freq = self._tr_01.freq
        self._pulse01_schedule = self._tr_01.sweep_schedule
        super().__init__(duration)

    def tr_create_pulse(self):
        """

        :return:
        """
        est_freq = self._pulse01_freq + 1 * ANHAR
        max_freq, min_freq = est_freq + 30 * 1e6, est_freq - 30 * 1e6
        self._freq_probe = linspace(min_freq, max_freq, 100)
        self._freq = Parameter("freq")

        # Setting up pulse properties for 12
        with pulse.build(backend=backend) as sweep_schedule:
            drive_chan = pulse.drive_channel(qubit=QUBIT_VAL)
            pulse.delay(48, drive_chan)
            pulse.set_frequency(self.freq, drive_chan)
            pulse.play(pulse.Gaussian(
                duration=self.duration,  # input
                amp=0.2,
                sigma=self.duration / 4,
            ), drive_chan)
            self._sweep_schedule = sweep_schedule

        # If the system does not respond -> TimeoutError occurs
        if not self._sweep_schedule:
            raise TimeoutError

    def tr_create_gate(self):
        """

        :return:
        """
        g01 = Gate("g01", 1, [])
        g12 = Gate("g12", 1, [self.freq])
        qc = QuantumCircuit(1, 1)
        qc.append(g01, [0])
        qc.append(g12, [0])
        qc.measure(0, 0)
        qc.add_calibration(g01, (0,), self._pulse01_schedule, [self.freq])
        qc.add_calibration(g12, (0,), self._sweep_schedule, [self.freq])

        # Get the circuits from assigned frequencies
        self._exp_circuits = [qc.assign_parameters({self.freq: f}, inplace=False) for f in self._freq_probe]

    def run_monitor(self):
        """"""
        pass

