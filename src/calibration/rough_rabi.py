"""Rough rabi techniques"""
from qiskit.tools import job_monitor

from src.calibration.calibration_utility import g01, g12
from abc import ABC, abstractmethod
from qiskit.circuit import Gate, QuantumCircuit
from qiskit import schedule
from src.calibration import backend
import numpy as np
from src.constant import QUBIT_PARA


class Rough_Rabi:
    """
    The class act as provider + regulator for Rough Rabi techniques
    """
    def __init__(self) -> None:
        """

        """
        self._package = list()
        self.amps = np.linspace(-0.75, 0.75, 100)
        self._job_id = 0
        # Standard flow of protocol
        self.rr_circuit()
        self.rr_job_monitor()

    @property
    def job_id(self):
        return self._job_id

    @abstractmethod
    def rr_circuit(self):
        raise NotImplementedError

    @abstractmethod
    def rr_job_monitor(self):
        raise NotImplementedError


class Rough_Rabi01(Rough_Rabi):
    """

    """
    def __init__(self) -> None:
        """

        """
        super().__init__()

    def rr_circuit(self):
        """

        :return:
        """
        right_theta_01 = Gate(r'$\theta_x^{01}$', 1, [])
        for a in self.amps:
            amp01 = QuantumCircuit(1, 1)
            amp01.append(right_theta_01, [0])
            amp01.measure(0, 0)
            amp01.add_calibration(right_theta_01, (0,), g01(a, 0), [])
            self._package.append(amp01)

    def rr_job_monitor(self):
        """

        :return:
        """
        schedule(self._package[99], backend=backend).draw(backend=backend)
        amp01_job = backend.run(self._package, meas_level=1, meas_return='avg', shots=2048)
        self._job_id = amp01_job.job_id()
        job_monitor(amp01_job)


class Rough_Rabi12(Rough_Rabi):
    """

    """
    def __init__(self) -> None:
        """

        """
        super().__init__()

    def rr_circuit(self) -> None:
        """

        :return:
        """
        theta_12_x = Gate(r'$\theta_x^{(12)}$', 1, [])
        pi_01_x = Gate(r'$\pi_x^{(01)}$', 1, [])

        for a in self.amps:
            amp12 = QuantumCircuit(1, 1)
            amp12.append(pi_01_x, [0])
            amp12.append(theta_12_x, [0])
            amp12.measure(0, 0)
            amp12.add_calibration(pi_01_x, (0,), g01(QUBIT_PARA.PI.value, 0), [])
            amp12.add_calibration(theta_12_x, (0,), g12(a, 0), [])
            self._package.append(amp12)

    def rr_job_monitor(self) -> None:
        """

        :return:
        """
        schedule(self._package[0], backend=backend).draw()
        amp12_job = backend.run(self._package, meas_level=1, meas_return='avg', shots=2 ** 14)
        self._job_id = amp12_job.job_id()
        job_monitor(amp12_job)
