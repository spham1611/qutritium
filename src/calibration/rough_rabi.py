"""Rough rabi techniques"""
from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.tools.monitor import job_monitor
from src.utility import fit_function
from src.calibration import (
    backend,
    QUBIT_VAL
)
from src.calibration.calibration_utility import Gate_Schedule
from src.pulse import Pulse01, Pulse12
from src.analyzer import DataAnalysis
from src.constant import QUBIT_PARA
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union
import numpy as np


class Rough_Rabi(ABC):
    """
    The class act as provider + regulator for Rough Rabi techniques.
    Rough Rabi flow: Set up -> create circuit -> submit job to IBM -> get the result
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12], num_shots: int) -> None:
        """

        :param pulse_model: Incomplete pulse: duration + freq
        """
        self.pulse_model = pulse_model
        self.num_shots = num_shots

        self.x_amp = None
        self.submitted_job = None
        self.package: Optional[List] = None

        # INTERNAL DESIGN ONLY
        self._lambda_list: Optional[List] = None
        self._x_amp_sweeping_range = np.linspace(-1, 1, 100)

    @property
    def lambda_list(self) -> List[float]:
        return self._lambda_list

    @lambda_list.setter
    def lambda_list(self, val_list: list) -> None:
        if len(val_list) != 4:
            raise ValueError("Lambda list does not have sufficient elements")
        self._lambda_list = val_list

    def run(self) -> None:
        """
        Standard RR01 protocol
        :return:
        """
        self.rr_create_circuit()
        self.rr_job_monitor()
        self.modify_pulse_model()
        print("Process run successfully!")

    @abstractmethod
    def rr_create_circuit(self) -> None:
        raise NotImplementedError

    def modify_pulse_model(self) -> None:
        raise NotImplementedError

    def rr_job_monitor(self) -> None:
        """

        :return:
        """
        self.submitted_job = backend.run(self.package,
                                         meas_level=1,
                                         meas_return='avg',
                                         shots=self.num_shots)
        job_monitor(self.submitted_job)

    def analyze(self, job_id: str = "") -> float:
        """

        :return:
        """
        if job_id is None:
            analyzer = DataAnalysis(experiment=self.submitted_job, num_shots=self.num_shots)
        else:
            experiment = backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)

        analyzer.retrieve_data(average=True)
        fit_params, _ = fit_function(self._x_amp_sweeping_range, analyzer.IQ_data,
                                     lambda x, c1, c2, drive_period, phi:
                                     (c1 * np.cos(2 * np.pi * x / drive_period - phi) + c2),
                                     [5, 0, 0.5, 0])
        x_amp = (fit_params[2] / 2)
        return x_amp


class Rough_Rabi01(Rough_Rabi):
    """

    """

    def __init__(self, pulse_model: Pulse01, num_shots: int = 20000) -> None:
        """

        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = [5, 0, 0.5, 0]
        self.x_amp = Parameter('x01_amp')

    def run(self) -> None:
        """

        :return:
        """
        super().run()

    def rr_create_circuit(self) -> None:
        """

        :return:
        """
        x01_gate = Gate('Unitary', 1, [self.x_amp])
        qc_rabi01 = QuantumCircuit(7, 1)
        qc_rabi01.append(x01_gate, [QUBIT_VAL])
        qc_rabi01.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        qc_rabi01.add_calibration(x01_gate, [QUBIT_VAL],
                                  Gate_Schedule.single_gate_schedule(self.pulse_model.frequency,
                                                                     self.pulse_model.duration,
                                                                     self.x_amp,
                                                                     ),
                                  [self.x_amp])
        self.package = [qc_rabi01.assign_parameters({self.x_amp: a}, inplace=False)
                        for a in self._x_amp_sweeping_range]

    def modify_pulse_model(self) -> None:
        """

        :return:
        """
        self.pulse_model.x_amp = self.analyze()


class Rough_Rabi12(Rough_Rabi):
    """

    """

    def __init__(self, pulse_model: Pulse12, num_shots: int = 20000) -> None:
        """
        Assume we have amp_x in our pulse model
        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = []
        self.x_amp = Parameter('x12_amp')

    def run(self) -> None:
        """

        :return:
        """
        super().run()

    def rr_create_circuit(self) -> None:
        """

        :return:
        """
        pass

    def modify_pulse_model(self) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse12
        self.pulse_model.x_amp = self.analyze()

