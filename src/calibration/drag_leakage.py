"""Drag Leakage classes and their techniques"""
from qiskit.circuit import Parameter, Gate, QuantumCircuit
from qiskit.tools.monitor import job_monitor
from qiskit.pulse.schedule import ScheduleBlock
from src.analyzer import DataAnalysis
from src.pulse import Pulse01, Pulse12
from src.calibration import backend, QUBIT_VAL
from src.calibration.calibration_utility import Gate_Schedule
from src.exceptions.pulse_exception import MissingDurationPulse, MissingFrequencyPulse, MissingAmplitudePulse
# from src.utility import fit_function
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import qiskit.pulse as pulse
import numpy as np


class DragLK(ABC):
    """
    The class act as provider + regulator for Drag Leakage.
    Drag Leakage flow: set up -> create circuit -> submit job to IBM -> get the result
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12], num_shots: int) -> None:
        """

        :param pulse_model:
        :param num_shots:
        :return:
        """

        if pulse_model.duration == 0:
            raise MissingDurationPulse
        if pulse_model.frequency == 0:
            raise MissingFrequencyPulse
        if pulse_model.x_amp == 0:
            raise MissingAmplitudePulse

        self.pulse_model = pulse_model
        self.num_shots = num_shots
        self.submitted_job_id = None
        self.package: Optional[List[QuantumCircuit]] = None
        self.x01_gate: Optional[Gate] = None
        self.x12_gate: Optional[Gate] = None
        self.xx_gate: Optional[Gate] = None
        self.xx_schedule: Optional[ScheduleBlock] = None

        # INTERNAL DESIGN ONLY
        self.drive_betas = np.linspace(-5, 5, 32)
        self.drive_beta = Parameter('drive_beta')

    @abstractmethod
    def x01_schedule(self) -> ScheduleBlock:
        """
        Helper function, mainly used for discriminator
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def x12_schedule(self) -> ScheduleBlock:
        """
        Helper function, mainly used for discriminator
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def append_circuit(self, number_append: int) -> List[QuantumCircuit]:
        """

        :param number_append:
        :return:
        """
        raise NotImplementedError

    def establish_discriminator(self) -> List[QuantumCircuit]:
        """

        :return:
        """
        self.x01_gate = Gate(r'X^{01}', 1, [])
        self.x12_gate = Gate(r'X^{12}', 1, [])

        ground_state = QuantumCircuit(7, 1)
        ground_state.measure(QUBIT_VAL, 0)

        first_excited_state = QuantumCircuit(7, 1)
        first_excited_state.append(self.x01_gate, [QUBIT_VAL])
        first_excited_state.measure(QUBIT_VAL, 0)
        first_excited_state.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])

        second_excited_state = QuantumCircuit(7, 1)
        second_excited_state.append(self.x01_gate, [QUBIT_VAL])
        second_excited_state.append(self.x12_gate, [QUBIT_VAL])
        second_excited_state.measure(QUBIT_VAL, 0)
        second_excited_state.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])
        second_excited_state.add_calibration(self.x12_gate, (QUBIT_VAL,), self.x12_schedule(), [])

        return [ground_state, first_excited_state, second_excited_state]

    def analyze(self, job_id: str, index_taken: int = 0) -> float:
        """

        :param job_id:
        :param index_taken:
        :return:
        """
        if job_id is None:
            experiment = backend.retrieve_job(self.submitted_job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)
        else:
            experiment = backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)

        # Analyze process
        analyzer.retrieve_data(average=False)
        analyzer.lda()
        analyzer.count_pop()
        analyzer.error_mitiq()
        drag_values_n7 = analyzer.mitiq_data[67:99, index_taken]
        beta = self.drive_betas[np.argmax(drag_values_n7)]
        return beta

    def run(self) -> None:
        """

        :return:
        """
        self.set_up()
        self.dl_create_circuit()
        self.run_monitor()
        self.modify_pulse_model()
        print("Process run successfully!")

    @abstractmethod
    def set_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def modify_pulse_model(self, job_id: str = None) -> None:
        raise NotImplementedError

    def dl_create_circuit(self) -> None:
        """

        :return:
        """
        exp_drag_circuit = self.append_circuit(number_append=3) \
                           + self.append_circuit(number_append=5) \
                           + self.append_circuit(number_append=7)
        self.package = self.establish_discriminator() + exp_drag_circuit

    def run_monitor(self) -> None:
        """

        :return:
        """
        submitted_job = backend.run(self.package,
                                    meas_level=1,
                                    meas_return='single',
                                    shots=self.num_shots)
        self.submitted_job_id = submitted_job.job_id()
        job_monitor(submitted_job)


class DragLK01(DragLK):
    """

    """

    def __init__(self, pulse_model: Pulse01, num_shots: int = 20000) -> None:
        """
        Pulse01 must have pulse12 because we are calculating the leakage coming from 12
        :param pulse_model:
        :param num_shots:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)

    def x01_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse01
        x01_sch = Gate_Schedule.single_gate_schedule(
            self.pulse_model.frequency,
            self.pulse_model.duration,
            self.pulse_model.x_amp,
        )
        return x01_sch

    def x12_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse01
        x12_sch = Gate_Schedule.single_gate_schedule(
            self.pulse_model.pulse12.frequency,
            self.pulse_model.pulse12.duration,
            self.pulse_model.pulse12.x_amp
        )
        return x12_sch

    def append_circuit(self, number_append: int) -> List[QuantumCircuit]:
        """
        Append xx gate repeatedly
        :param number_append:
        :return:
        """
        qc_drag = QuantumCircuit(7, 1)
        for _ in range(number_append):
            qc_drag.append(self.xx_gate, [QUBIT_VAL])
        qc_drag.measure(QUBIT_VAL, 0)
        qc_drag.add_calibration(self.xx_gate, (QUBIT_VAL,), self.xx_schedule, [self.drive_beta])
        return [qc_drag.assign_parameters({self.drive_beta: b}, inplace=False) for b in self.drive_betas]

    def set_up(self) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse01
        with pulse.build(backend=backend) as xx_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(self.pulse_model.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(duration=self.pulse_model.duration,
                                  amp=self.pulse_model.pulse12.x_amp,
                                  sigma=self.pulse_model.duration / 4,
                                  beta=self.drive_beta), drive_chan)
            pulse.play(pulse.Drag(duration=self.pulse_model.duration,
                                  amp=-self.pulse_model.pulse12.x_amp,
                                  sigma=self.pulse_model.duration / 4,
                                  beta=self.drive_beta), drive_chan)

        self.xx_schedule = xx_schedule
        self.xx_gate = Gate("$X_\pi X_{-\pi}$", 1, [self.drive_beta])

    def modify_pulse_model(self, job_id: str = None) -> None:
        """
        Add beta leakage to pulse model 01
        :return:
        """
        self.pulse_model: Pulse01
        self.pulse_model.beta_leakage = self.analyze(job_id=job_id)


class DragLK12(DragLK):
    """

    """

    def __init__(self, pulse_model: Pulse12, num_shots=20000) -> None:
        """
        Pulse12 must have its pulse01
        :param pulse_model:
        :param num_shots:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)

    def x01_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse12
        x01_sch = Gate_Schedule.single_gate_schedule(
            self.pulse_model.pulse01.frequency,
            self.pulse_model.pulse01.duration,
            self.pulse_model.pulse01.x_amp,
        )
        return x01_sch

    def x12_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse12
        x12_sch = Gate_Schedule.single_gate_schedule(
            self.pulse_model.frequency,
            self.pulse_model.duration,
            self.pulse_model
        )
        return x12_sch

    def append_circuit(self, number_append: int) -> List[QuantumCircuit]:
        """
        Append xx gate repeatedly
        :param number_append:
        :return:
        """
        qc_drag = QuantumCircuit(7, 1)
        qc_drag.append(self.x01_gate, [QUBIT_VAL])
        for _ in range(number_append):
            qc_drag.append(self.xx_gate, [QUBIT_VAL])

        qc_drag.measure(QUBIT_VAL, 0)
        qc_drag.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])
        qc_drag.add_calibration(self.xx_gate, (QUBIT_VAL,), self.xx_schedule, [self.drive_beta])
        return [qc_drag.assign_parameters({self.drive_beta: b}, inplace=False) for b in self.drive_betas]

    def set_up(self) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse12
        with pulse.build(backend=backend) as xx_schedule:
            drive_chan = pulse.drive_channel(QUBIT_VAL)
            pulse.set_frequency(self.pulse_model.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(duration=self.pulse_model.duration,
                                  amp=self.pulse_model.x_amp,
                                  sigma=self.pulse_model.duration / 4,
                                  beta=self.drive_beta), drive_chan)
            pulse.play(pulse.Drag(duration=self.pulse_model.duration,
                                  amp=-self.pulse_model.x_amp,
                                  sigma=self.pulse_model.duration / 4,
                                  beta=self.drive_beta), drive_chan)
        self.xx_schedule = xx_schedule
        self.xx_gate = Gate("$X_\pi X_{-\pi}$", 1, [self.drive_beta])

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse12
        self.pulse_model.beta_leakage = self.analyze(job_id=job_id, index_taken=1)
