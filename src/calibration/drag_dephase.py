"""Drag Dephase classes and their techniques"""
import qiskit.pulse as pulse
import numpy as np
import pandas as pd
from qiskit.circuit import Parameter, Gate, QuantumCircuit
from qiskit.tools.monitor import job_monitor
from qiskit.pulse.schedule import ScheduleBlock
from src.analyzer import DataAnalysis
from src.pulse import Pulse01, Pulse12
from src.calibration import backend, provider, QUBIT_VAL
from src.pulse_creation import Gate_Schedule
from src.utility import fit_function
from abc import ABC, abstractmethod
from typing import List, Union, Optional


class DragDP(ABC):
    """
    The class act as provider + regulator for Drag Dephase techniques.
    Drag flow: set up -> create circuit -> submit job to IBM -> get the result
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12], num_shots: int) -> None:
        """

        :param pulse_model:
        :param num_shots:
        """

        # if pulse_model.duration == 0:
        #     raise MissingDurationPulse
        # if pulse_model.frequency == 0:
        #     raise MissingFrequencyPulse
        # if pulse_model.x_amp == 0:
        #     raise MissingAmplitudePulse
        # Missing pulse12 check

        self.pulse_model = pulse_model
        self.num_shots = num_shots
        self.submitted_job_id = None
        self.package: list[QuantumCircuit] = []
        self.x01_gate: Optional[Gate] = None
        self.x12_gate: Optional[Gate] = None
        self.drag_inst_x: Optional[ScheduleBlock] = None
        self.drag_inst_yp: Optional[ScheduleBlock] = None
        self.drag_inst_ym: Optional[ScheduleBlock] = None
        self.discriminator: list[QuantumCircuit] = []

        # Internal design only
        self.drive_beta = Parameter('drive_beta')
        self.drive_betas = np.linspace(-5, 5, 32)
        self._lambda_list = [0., 0.5]

    @property
    def lambda_list(self) -> List:
        return self._lambda_list

    @lambda_list.setter
    def lambda_list(self, val_list: List) -> None:
        if len(val_list) != 2:
            raise ValueError("Lambda list does not have sufficient elements")
        self._lambda_list = val_list

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

    def save_data(self, drag_x, drag_yp, drag_ym) -> None:
        """

        :param drag_x:
        :param drag_yp:
        :param drag_ym:
        :return:
        """
        data = {
            'x_val': self.drive_betas,
            'drag_x': drag_x,
            'drag_yp': drag_yp,
            'drag_ym': drag_ym,
        }
        df = pd.DataFrame(data)
        df.to_csv(f'Drag_Dephase of {self.pulse_model.__class__.__name__}.csv', index=False)

    def establish_discriminator(self) -> None:
        """
        Create excited circuit -> add to final circuit
        :return:
        """
        self.x01_gate = Gate(r'X^{01}', 1, [])
        self.x12_gate = Gate(r'X^{12}', 1, [])

        ground_state = QuantumCircuit(QUBIT_VAL + 1, 1)
        ground_state.measure(QUBIT_VAL, 0)

        # Circuit for first excited state
        first_excited_state = QuantumCircuit(QUBIT_VAL + 1, 1)
        first_excited_state.append(self.x01_gate, [QUBIT_VAL])
        first_excited_state.measure(QUBIT_VAL, 0)
        first_excited_state.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])

        # Circuit for second excited state
        second_excited_state = QuantumCircuit(QUBIT_VAL + 1, 1)
        second_excited_state.append(self.x01_gate, [QUBIT_VAL])
        second_excited_state.append(self.x12_gate, [QUBIT_VAL])
        second_excited_state.measure(QUBIT_VAL, 0)
        second_excited_state.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])
        second_excited_state.add_calibration(self.x12_gate, (QUBIT_VAL,), self.x12_schedule(), [])

        self.discriminator += [ground_state, first_excited_state, second_excited_state]

    def analyze(self, job_id: str, index_taken: int = 0) -> float:
        """

        :return:
        """
        if job_id is None:
            experiment = provider.backend.retrieve_job(self.submitted_job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)
        else:
            experiment = provider.backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)

        # Analyze process
        analyzer.retrieve_data(average=False)
        analyzer.lda()
        analyzer.count_pop()
        analyzer.error_mitiq()

        drag_values_x = analyzer.mitiq_data[3:35, 0]
        drag_values_yp = analyzer.mitiq_data[35:67, index_taken]
        drag_values_ym = analyzer.mitiq_data[67:99, index_taken]
        self.save_data(drag_x=drag_values_x,
                       drag_yp=drag_values_yp,
                       drag_ym=drag_values_ym)

        fit_params_yp, y_fit_yp = fit_function(self.drive_betas,
                                               drag_values_yp,
                                               lambda x, c1, c2: c1 * x + c2,
                                               self.lambda_list)
        fit_params_ym, y_fit_ym = fit_function(self.drive_betas,
                                               drag_values_ym,
                                               lambda x, c1, c2: c1 * x + c2,
                                               self.lambda_list)

        # Extract beta
        mat = np.array([
            [fit_params_yp[0], -1.0],
            [fit_params_ym[0], -1.0]
        ])
        y_vec = np.array([-fit_params_yp[1], -fit_params_ym[1]])
        beta = np.linalg.solve(mat, y_vec)
        return beta[0]

    def run(self) -> None:
        """

        :return:
        """
        self.establish_discriminator()
        self.set_up()
        self.dp_create_circuit()
        self.run_monitor()
        self.modify_pulse_model()
        print("Process run successfully!")

    @abstractmethod
    def set_up(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def dp_create_circuit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def modify_pulse_model(self, job_id: str = None) -> None:
        raise NotImplementedError

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


class DragDP01(DragDP):
    def __init__(self, pulse_model: Pulse01, num_shots: int = 20000) -> None:
        """

        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)

    def run(self) -> None:
        """

        :return:
        """
        super().run()

    def x01_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse01
        x01_sch = Gate_Schedule.single_gate_schedule_gaussian(
            drive_freq=self.pulse_model.frequency,
            drive_duration=self.pulse_model.duration,
            drive_amp=self.pulse_model.x_amp,
        )
        return x01_sch

    def x12_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse01
        x12_sch = Gate_Schedule.single_gate_schedule_gaussian(
            drive_freq=self.pulse_model.pulse12.frequency,
            drive_duration=self.pulse_model.pulse12.duration,
            drive_amp=self.pulse_model.pulse12.x_amp,
            name='$X^{12}$'
        )
        return x12_sch

    def set_up(self) -> None:
        """

        :return:
        """

        # Create ...
        self.pulse_model: Pulse01
        with pulse.build(backend=backend) as drag_inst_x:
            pulse.set_frequency(self.pulse_model.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.duration, self.pulse_model.x_amp / 2,
                                  self.pulse_model.duration / 4, self.drive_beta, name=r'$\pi/2_x$'),
                       pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.duration, self.pulse_model.x_amp,
                                  self.pulse_model.duration / 4, self.drive_beta, name=r'$\pi_x$'),
                       pulse.drive_channel(QUBIT_VAL))
        self.drag_inst_x = drag_inst_x

        # Create ...
        with pulse.build(backend=backend) as drag_inst_yp:
            pulse.set_frequency(self.pulse_model.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.duration, self.pulse_model.x_amp / 2,
                                  self.pulse_model.duration / 4, self.drive_beta, name=r'$\pi/2_x$'),
                       pulse.drive_channel(QUBIT_VAL))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.duration, self.pulse_model.x_amp,
                                  self.pulse_model.duration / 4, self.drive_beta, name=r'$\pi_x$'),
                       pulse.drive_channel(QUBIT_VAL))
        self.drag_inst_yp = drag_inst_yp

        # Create ...
        with pulse.build(backend=backend) as drag_inst_ym:
            pulse.set_frequency(self.pulse_model.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.duration, self.pulse_model.x_amp / 2,
                                  self.pulse_model.duration / 4, self.drive_beta, name=r'$\pi/2_x$'),
                       pulse.drive_channel(QUBIT_VAL))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.duration, -self.pulse_model.x_amp,
                                  self.pulse_model.duration / 4, self.drive_beta, name=r'$-\pi_y$'),
                       pulse.drive_channel(QUBIT_VAL))
        self.drag_inst_ym = drag_inst_ym

    def dp_create_circuit(self):
        """

        :return:
        """
        self.pulse_model: Pulse01
        drag_gate_x = Gate(r'$\pi/2_x\cdot\pi_x$', 1, [self.drive_beta])
        drag_circ = QuantumCircuit(QUBIT_VAL + 1, 1)
        drag_circ.append(drag_gate_x, [QUBIT_VAL])
        drag_circ.measure(QUBIT_VAL, 0)
        drag_circ.add_calibration(drag_gate_x, (QUBIT_VAL,), self.drag_inst_x, [self.drive_beta])
        drag_circuit_x = [drag_circ.assign_parameters({self.drive_beta: b}, inplace=False)
                          for b in self.drive_betas]

        drag_gate_yp = Gate(r'$\pi/2_x\cdot\pi_y$', 1, [self.drive_beta])
        drag_circ = QuantumCircuit(QUBIT_VAL + 1, 1)
        drag_circ.append(drag_gate_yp, [QUBIT_VAL])
        drag_circ.measure(QUBIT_VAL, 0)
        drag_circ.add_calibration(drag_gate_yp, (QUBIT_VAL,), self.drag_inst_yp, [self.drive_beta])
        drag_circuit_yp = [drag_circ.assign_parameters({self.drive_beta: b}, inplace=False)
                           for b in self.drive_betas]

        drag_gate_ym = Gate(r'$\pi/2_x\cdot -\pi_y$', 1, [self.drive_beta])
        drag_circ = QuantumCircuit(QUBIT_VAL + 1, 1)
        drag_circ.append(drag_gate_ym, [QUBIT_VAL])
        drag_circ.measure(QUBIT_VAL, 0)
        drag_circ.add_calibration(drag_gate_ym, (QUBIT_VAL,), self.drag_inst_ym, [self.drive_beta])
        drag_circuit_ym = [drag_circ.assign_parameters({self.drive_beta: b}, inplace=False)
                           for b in self.drive_betas]

        self.package = self.discriminator + drag_circuit_x + drag_circuit_yp + drag_circuit_ym

    def modify_pulse_model(self, job_id: str = None) -> None:
        """
        Add beta to pulse_model (01)
        :return:
        """
        self.pulse_model: Pulse01
        self.pulse_model.beta_dephase = self.analyze(job_id=job_id)


class DragDP12(DragDP):
    def __init__(self, pulse_model: Pulse12, num_shots: int = 20000) -> None:
        """

        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)

    def run(self) -> None:
        """

        :return:
        """
        super().run()

    def x01_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse12
        x01_sch = Gate_Schedule.single_gate_schedule_gaussian(
            drive_freq=self.pulse_model.pulse01.frequency,
            drive_duration=self.pulse_model.pulse01.duration,
            drive_amp=self.pulse_model.x_amp,
        )
        return x01_sch

    def x12_schedule(self) -> ScheduleBlock:
        """

        :return:
        """
        self.pulse_model: Pulse12
        x12_sch = Gate_Schedule.single_gate_schedule_gaussian(
            drive_freq=self.pulse_model.frequency,
            drive_duration=self.pulse_model.duration,
            drive_amp=self.pulse_model.x_amp,
            name='$X^{12}$'
        )
        return x12_sch

    def set_up(self) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse12
        with pulse.build(backend=backend) as drag_inst_x:
            pulse.set_frequency(self.pulse_model.pulse01.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.pulse01.duration, self.pulse_model.pulse01.x_amp / 2,
                                  self.pulse_model.pulse01.duration / 4, self.drive_beta, name=r'$\pi/2_x$'),
                       pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.pulse01.duration, self.pulse_model.pulse01.x_amp,
                                  self.pulse_model.pulse01.duration / 4, self.drive_beta, name=r'$\pi_x$'),
                       pulse.drive_channel(QUBIT_VAL))
        self.drag_inst_x = drag_inst_x

        with pulse.build(backend=backend) as drag_inst_yp:
            pulse.set_frequency(self.pulse_model.pulse01.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.pulse01.duration, self.pulse_model.pulse01.x_amp / 2,
                                  self.pulse_model.pulse01.duration / 4, self.drive_beta, name=r'$\pi/2_x$'),
                       pulse.drive_channel(QUBIT_VAL))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.pulse01.duration, self.pulse_model.pulse01.x_amp,
                                  self.pulse_model.pulse01.duration / 4, self.drive_beta, name=r'$\pi_y$'),
                       pulse.drive_channel(QUBIT_VAL))
        self.drag_inst_yp = drag_inst_yp

        with pulse.build(backend=backend) as drag_inst_ym:
            pulse.set_frequency(self.pulse_model.pulse01.frequency, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.pulse01.duration, self.pulse_model.pulse01.x_amp / 2,
                                  self.pulse_model.pulse01.duration / 4, self.drive_beta, name=r'$\pi/2_x$'),
                       pulse.drive_channel(QUBIT_VAL))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(QUBIT_VAL))
            pulse.play(pulse.Drag(self.pulse_model.pulse01.duration, -self.pulse_model.pulse01.x_amp,
                                  self.pulse_model.pulse01.duration / 4, self.drive_beta, name=r'$-\pi_y$'),
                       pulse.drive_channel(QUBIT_VAL))
        self.drag_inst_ym = drag_inst_ym

    def dp_create_circuit(self) -> None:
        """

        :return:
        """
        drag_gate_x = Gate(r'$\pi/2_x\cdot\pi_x$', 1, [self.drive_beta])
        drag_circ = QuantumCircuit(QUBIT_VAL + 1, 1)
        drag_circ.append(self.x01_gate, [QUBIT_VAL])
        drag_circ.append(drag_gate_x, [QUBIT_VAL])
        drag_circ.measure(QUBIT_VAL, 0)
        drag_circ.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])
        drag_circ.add_calibration(drag_gate_x, (QUBIT_VAL,), self.drag_inst_x, [self.drive_beta])
        drag_circ_x = [drag_circ.assign_parameters({self.drive_beta: b}, inplace=False) for b in self.drive_betas]

        drag_gate_yp = Gate(r'$\pi/2_x\cdot\pi_y$', 1, [self.drive_beta])
        drag_circ = QuantumCircuit(QUBIT_VAL + 1, 1)
        drag_circ.append(self.x01_gate, [QUBIT_VAL])
        drag_circ.append(drag_gate_yp, [QUBIT_VAL])
        drag_circ.measure(QUBIT_VAL, 0)
        drag_circ.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])
        drag_circ.add_calibration(drag_gate_yp, (QUBIT_VAL,), self.drag_inst_yp, [self.drive_beta])
        drag_circ_yp = [drag_circ.assign_parameters({self.drive_beta: b}, inplace=False) for b in self.drive_betas]

        drag_gate_ym = Gate(r'$\pi/2_x\cdot -\pi_y$', 1, [self.drive_beta])
        drag_circ = QuantumCircuit(QUBIT_VAL + 1, 1)
        drag_circ.append(self.x01_gate, [QUBIT_VAL])
        drag_circ.append(drag_gate_ym, [QUBIT_VAL])
        drag_circ.measure(QUBIT_VAL, 0)
        drag_circ.add_calibration(self.x01_gate, (QUBIT_VAL,), self.x01_schedule(), [])
        drag_circ.add_calibration(drag_gate_ym, (QUBIT_VAL,), self.drag_inst_ym, [self.drive_beta])
        drag_circ_ym = [drag_circ.assign_parameters({self.drive_beta: b}, inplace=False) for b in self.drive_betas]

        self.package = self.discriminator + drag_circ_x + drag_circ_yp + drag_circ_ym

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        :return:
        """
        self.pulse_model: Pulse12
        self.pulse_model.beta_dephase = self.analyze(job_id=job_id, index_taken=1)
