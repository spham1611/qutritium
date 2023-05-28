"""Rough rabi techniques"""
import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.tools.monitor import job_monitor
from src.utility import fit_function
from src.calibration import (
    backend,
    provider,
    QUBIT_VAL
)
from src.pulse_creation import Gate_Schedule
from src.pulse import Pulse01, Pulse12
from src.analyzer import DataAnalysis
from src.constant import QUBIT_PARA
from src.exceptions.pulse_exception import MissingDurationPulse, MissingFrequencyPulse
from abc import ABC, abstractmethod
from typing import Optional, List, Union


class Rough_Rabi(ABC):
    """
    The class act as provider + regulator for Rough Rabi techniques.
    Rough Rabi flow: Set up -> create circuit -> submit job to IBM -> get the result
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12], num_shots: int) -> None:
        """

        :param pulse_model: Incomplete pulse: Must have duration + freq
        :param num_shots:
        """

        if pulse_model.duration == 0:
            raise MissingDurationPulse
        if pulse_model.frequency == 0:
            raise MissingFrequencyPulse

        self.pulse_model = pulse_model
        self.num_shots = num_shots

        self.x_amp = None
        self.submitted_job_id = None
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

    def modify_pulse_model(self, job_id: str = None) -> None:
        raise NotImplementedError

    def save_data(self, y_values) -> None:
        """
        Save as csv
        :param y_values:
        :return:
        """
        data = {
            'x_amp': self._x_amp_sweeping_range,
            'x_amp_val': y_values
        }
        df = pd.DataFrame(data)
        df.to_csv(f'Rabii of {self.pulse_model.__class__.__name__}.csv', index=False)

    def rr_job_monitor(self) -> None:
        """

        :return:
        """
        submitted_job = backend.run(self.package,
                                    meas_level=1,
                                    meas_return='avg',
                                    shots=self.num_shots)
        self.submitted_job_id = submitted_job.job_id()
        job_monitor(submitted_job)

    def analyze(self, job_id: str) -> float:
        """

        :return:
        """
        if job_id is None:
            experiment = provider.backend.retrieve_job(self.submitted_job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)
        else:
            experiment = provider.backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)

        analyzer.retrieve_data(average=True)
        fit_params, _ = fit_function(self._x_amp_sweeping_range, analyzer.IQ_data,
                                     lambda x, c1, c2, drive_period, phi:
                                     (c1 * np.cos(2 * np.pi * x / drive_period - phi) + c2),
                                     [5, 0, 0.5, 0])
        self.save_data(analyzer.IQ_data)

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
        self.pulse_model: Pulse01
        x01_gate = Gate('Unitary', 1, [self.x_amp])
        qc_rabi01 = QuantumCircuit(QUBIT_VAL + 1, 1)
        qc_rabi01.append(x01_gate, [QUBIT_VAL])
        qc_rabi01.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        qc_rabi01.add_calibration(x01_gate, [QUBIT_VAL],
                                  Gate_Schedule.single_gate_schedule(drive_freq=self.pulse_model.frequency,
                                                                     drive_duration=self.pulse_model.duration,
                                                                     drive_amp=self.x_amp,
                                                                     ),
                                  [self.x_amp])
        self.package = [qc_rabi01.assign_parameters({self.x_amp: a}, inplace=False)
                        for a in self._x_amp_sweeping_range]

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse01
        self.pulse_model.x_amp = self.analyze(job_id=job_id)


class Rough_Rabi12(Rough_Rabi):
    """

    """

    def __init__(self, pulse_model: Pulse12, num_shots: int = 20000) -> None:
        """
        Assume we have amp_x in our pulse model
        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = [5, 0, 0.4, 0]
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
        self.pulse_model: Pulse12
        x01_pi = Gate(r'$X^{01}_\pi$', 1, [])
        x12_gate = Gate('Unitary', 1, [self.x_amp])
        qc_rabi12 = QuantumCircuit(QUBIT_VAL + 1, 1)
        qc_rabi12.append(x01_pi, [QUBIT_VAL])
        qc_rabi12.append(x12_gate, [QUBIT_VAL])
        qc_rabi12.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        qc_rabi12.add_calibration(x01_pi, [QUBIT_VAL],
                                  Gate_Schedule.single_gate_schedule(drive_freq=self.pulse_model.pulse01.frequency,
                                                                     drive_duration=self.pulse_model.pulse01.duration,
                                                                     drive_amp=self.pulse_model.pulse01.x_amp,
                                                                     ),
                                  [self.x_amp])
        qc_rabi12.add_calibration(x12_gate, [QUBIT_VAL],
                                  Gate_Schedule.single_gate_schedule(drive_freq=self.pulse_model.frequency,
                                                                     drive_duration=self.pulse_model.duration,
                                                                     drive_amp=self.x_amp,
                                                                     ),
                                  [self.x_amp])
        self.package = [qc_rabi12.assign_parameters({self.x_amp: a}, inplace=False)
                        for a in self._x_amp_sweeping_range]

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse12
        self.pulse_model.x_amp = self.analyze(job_id=job_id)
