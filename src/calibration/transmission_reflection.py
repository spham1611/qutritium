"""Transmission and reflection techniques"""
from abc import ABC, abstractmethod
from numpy import linspace
from . import backend
from ..constant import QUBIT_PARA
import numpy as np
from qiskit.circuit import Parameter, Gate, QuantumCircuit
import qiskit.pulse as pulse


class TR(ABC):
    """
    The class act as provider + regulator for transmission and reflection techniques
    """
    def __init__(self, duration=0) -> None:
        """

        :param duration:
        """
        self._duration = duration
        self._freq = 0
        self._sweep_schedule = None
        self._freq_probe = None
        self.tr_create_pulse()
        self.tr_create_gate()
        self.run_monitor()

    @property
    def duration(self):
        return self._duration
 
    @abstractmethod
    def tr_create_pulse(self):
        raise NotImplementedError

    @abstractmethod
    def tr_create_gate(self):
        raise NotImplementedError

    @abstractmethod
    def run_monitor(self):
        raise NotImplementedError


class TR_01(TR):
    """"""
    def __init__(self, duration) -> None:
        """

        :param duration:
        """
        super().__init__(duration)

    def tr_create_pulse(self):
        """

        :return:
        """
        est_freq = backend.configuration().hamiltonian['vars'][f'wq{QUBIT_PARA.QUBIT}']/(2*np.pi)
        max_freq, min_freq = est_freq + 30 * 1e6, est_freq - 30 * 1e6
        self._freq_probe = linspace(min_freq, max_freq, 100)
        self._freq = Parameter("freq")
        with pulse.build(backend=backend) as sweep_schedule:
            drive_chan = pulse.drive_channel(qubit=QUBIT_PARA.QUBIT)
            pulse.delay(48, drive_chan)
            pulse.set_frequency(self._freq, drive_chan)
            pulse.play(pulse.Gaussian(
                duration=self.duration,   # input
                amp=0.2,
                sigma=self.duration / 4,
            ), drive_chan)
        self._sweep_schedule = sweep_schedule

    def tr_create_gate(self):
        """

        :return:
        """
        g01 = Gate("g01", 1, self._freq)
        qc = QuantumCircuit(1, 1)
        qc.append(g01, [0])
        qc.measure(0, 0)
        qc.add_calibration(g01, (0, ), self._sweep_schedule, [self._freq])
        exp_circuits = [qc.assign_parameters({self._freq: f}, inplace=False) for f in self._freq_probe]

    def run_monitor(self):
        pass


class TR_12(TR):
    """"""
    def __init__(self, duration) -> None:
        """

        :param duration:
        """
        super().__init__(duration)

    def tr_create_pulse(self):
        """

        :return:
        """
        est_freq = pulse01freq + 1*anhar
        max_freq, min_freq = est_freq + 30 * 1e6, est_freq - 30 * 1e6
        self._freq_probe = linspace(min_freq, max_freq, 100)
        self._freq = Parameter("freq")
        with pulse.build(backend=backend) as sweep_schedule:
            drive_chan = pulse.drive_channel(qubit=QUBIT_PARA.QUBIT)
            pulse.delay(48, drive_chan)
            pulse.set_frequency(self._freq, drive_chan)
            pulse.play(pulse.Gaussian(
                duration=self.duration,  # input
                amp=0.2,
                sigma=self.duration / 4,
            ), drive_chan)
        self._sweep_schedule = sweep_schedule

    def tr_create_gate(self):
        """

        :return:
        """
        g01 = Gate("g01", 1, [])
        g12 = Gate("g12", 1, [self._freq])
        qc = QuantumCircuit(1, 1)
        qc.append(g01, [0])
        qc.append(g12, [0])
        qc.measure(0, 0)
        qc.add_calibration(g01, (0,), pulse01sched, [self._freq])
        qc.add_calibration(g12, (0,), self._sweep_schedule, [self._freq])
        exp_circuits = [qc.assign_parameters({self._freq: f}, inplace=False) for f in self._freq_probe]

    def run_monitor(self):
        """"""
        pass
