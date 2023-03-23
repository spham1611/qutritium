"""Transmission and reflection techniques for 0-1 and 1-2
In here TR stands for transmission and reflection
"""
from qiskit.circuit import Parameter, Gate, QuantumCircuit
from qiskit.tools.monitor import job_monitor
from typing import List, Optional, Union
from src.calibration import (
    backend,
    QUBIT_VAL,
    DEFAULT_F01,
    DEFAULT_F12
)
from src.pulse import Pulse01, Pulse12
from src.analyzer import DataAnalysis
from src.constant import QUBIT_PARA
from src.calibration.calibration_utility import Gate_Schedule
from abc import ABC, abstractmethod
from numpy import linspace
from src.utility import fit_function
import numpy as np


class TR(ABC):
    """
    The class act as provider + regulator for transmission and reflection techniques
    TR Process flow: set up pulse and gates -> submit job to IBM -> Analyze the resulted pulse
    (return Pulse model)
    """

    def __init__(self, pulse_model: Pulse01, num_shots: int) -> None:
        """
        Must have duration!
        :param pulse_model: input pulse which can be  01 or 12
        """
        if pulse_model.duration != 0:
            self.pulse_model = pulse_model
        else:
            raise ValueError("Can not establish process without duration parameter!")
        self.num_shots = num_shots
        self.frequency = None
        self.submitted_job = None
        self.freq_sweeping_range: Optional[List] = None
        self.freq_sweeping_range_ghz: Optional[List] = None
        self.package: Optional[List] = None

        # INTERNAL DESIGN ONLY
        self._lambda_list = [0, 0, 0, 0]

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
        Run standard TR protocol. Devs can change the oder if needed
        :return:
        """
        self.set_up()
        self.tr_create_circuit()
        self.run_monitor()
        self.modify_pulse_model()
        print("Process run successfully!")

    @abstractmethod
    def set_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def tr_create_circuit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def modify_pulse_model(self) -> None:
        raise NotImplementedError

    def run_monitor(self) -> None:
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

        :param job_id: Change if we want to use other job. Default = old job_d
        :return:
        """
        if job_id is None:
            analyzer = DataAnalysis(experiment=self.submitted_job, num_shots=self.num_shots)
        else:
            experiment = backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)

        analyzer.retrieve_data(average=True)

        fit_params, _ = fit_function(self.freq_sweeping_range, analyzer.IQ_data,
                                     lambda x, c1, q_freq, c2, c3:
                                     (c1 / np.pi) * (c2 / ((x - q_freq) ** 2 + c2 ** 2)) + c3,
                                     self.lambda_list)

        freq = fit_params[1] * QUBIT_PARA.GHZ.value
        return freq


class TR_01(TR):
    """"""

    def __init__(self, pulse_model: Pulse01, num_shots: int = 20000) -> None:
        """

        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = [10, 4.9, 1, -2]
        self.frequency = Parameter('transition_freq_01')

    def run(self) -> None:
        super().run()

    def set_up(self) -> None:
        """

        :return:
        """
        mhz_unit = QUBIT_PARA.MHZ.value
        max_freq, min_freq = DEFAULT_F01 + 36 * mhz_unit, DEFAULT_F01 - 36 * mhz_unit
        self.freq_sweeping_range = linspace(min_freq, max_freq, 100)
        self.freq_sweeping_range_ghz = self.freq_sweeping_range / QUBIT_PARA.GHZ.value

    def tr_create_circuit(self) -> None:
        """

        :return:
        """
        freq01_probe = Gate('Unitary', 1, [self.frequency])
        qc_spect01 = QuantumCircuit(7, 1)
        qc_spect01.append(freq01_probe, [QUBIT_VAL])
        qc_spect01.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        qc_spect01.add_calibration(freq01_probe, [QUBIT_VAL],
                                   Gate_Schedule.single_gate_schedule(self.frequency, 0,
                                                                      self.pulse_model.duration, 0.2,
                                                                      0),
                                   [self.frequency])

        # Get the circuits from assigned frequencies
        self.package = [qc_spect01.assign_parameters({self.frequency: f}, inplace=False)
                        for f in self.freq_sweeping_range]

    def modify_pulse_model(self):
        """

        :return:
        """

        # Add frequency to pulse01
        f01 = self.analyze()
        self.pulse_model.frequency = f01


class TR_12(TR):
    """"""

    def __init__(self, pulse_model: Pulse01, num_shots: int = 20000) -> None:
        """
        In this process, we create a new pulse12 -> Users don't need to create pulse12 from pulse12 class as there
        might be conflicts in pulse01 and 12 parameters
        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = [-10, 4.8, 1, -2]
        self.frequency = Parameter('transition_freq_12')
        self.pulse01_schedule = None

    def set_up(self) -> None:
        """

        :return:
        """
        self.pulse01_schedule = Gate_Schedule.single_gate_schedule(
            self.pulse_model.frequency,
            self.pulse_model.duration,
            self.pulse_model.x_amp,
        )
        mhz_unit = QUBIT_PARA.MHZ.value
        max_freq, min_freq = DEFAULT_F12 + 36 * mhz_unit, DEFAULT_F12 - 36 * mhz_unit
        self.freq_sweeping_range = np.linspace(min_freq, max_freq, 100)
        self.freq_sweeping_range_ghz = self.freq_sweeping_range / QUBIT_PARA.GHZ.value

    def tr_create_circuit(self) -> None:
        """

        :return:
        """
        freq12_probe = Gate('Unit', 1, [self.frequency])
        x01_pi_gate = Gate(r'$X^{01}_\pi$', 1, [])
        qc_spect12 = QuantumCircuit(7, 1)
        qc_spect12.append(x01_pi_gate, [QUBIT_VAL])
        qc_spect12.append(freq12_probe, [QUBIT_VAL])
        qc_spect12.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        qc_spect12.add_calibration(x01_pi_gate, [QUBIT_VAL], self.pulse01_schedule)
        qc_spect12.add_calibration(freq12_probe, [QUBIT_VAL],
                                   Gate_Schedule.single_gate_schedule(self.frequency, 0,
                                                                      self.pulse_model.duration, 0.2,
                                                                      0),
                                   [self.frequency])
        self.package = [qc_spect12.assign_parameters({self.frequency: f}, inplace=False)
                        for f in self.freq_sweeping_range]

    def modify_pulse_model(self) -> None:
        """

        :return:
        """

        # Create new pulse 12
        f12 = self.analyze()
        pulse12 = Pulse12(duration=self.pulse_model.duration,
                          frequency=f12)

        # Point address pulse01 and 12 to each other
        self.pulse_model.pulse12 = pulse12
        self.pulse_model.pulse12.pulse01 = self.pulse_model
