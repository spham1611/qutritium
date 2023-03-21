"""Drag techniques"""
from abc import ABC, abstractmethod
from typing import List
from qiskit.circuit import Parameter, Gate, QuantumCircuit
from src.pulse import Pulse
from src.calibration import backend
from src.calibration import ANHAR, DRIVE_FREQ
from src.calibration.calibration_utility import general_12_g01, drag_dephase12_g12
import qiskit.pulse as pulse
import numpy as np


class Drag(ABC):
    """
    The class act as provider + regulator for Drag techniques
    """

    def __init__(self, pulse_model: Pulse) -> None:
        """

        :param pulse_model:
        :return:
        """
        self.x_amp = pulse_model.x_amp
        self.amp_sx = pulse_model.sx_amp
        self.duration = pulse_model.duration

        # Instance of DRAG pulse model
        self.drag_inst_x = None
        self.drag_inst_yp = None
        self.drag_inst_ym = None

        # Drag Circuit
        self.drag_circuit = None

        # Other internal use parameter
        self._drive_beta = Parameter('drive_beta')
        self.job_id = ""

    def run(self) -> None:
        """

        :return:
        """
        self.drag_dep_pulse()
        self.drag_dep_create_circuit()
        self.run_monitor()
        self.analyze()

    @abstractmethod
    def drag_dep_pulse(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def drag_dep_create_circuit(self) -> None:
        raise NotImplementedError

    def run_monitor(self) -> None:
        """

        :return:
        """
        drag_job = backend.run(self.drag_circuit,
                               shots=2 ** 14)
        self.job_id = drag_job.job_id()

    def analyze(self) -> None:
        """

        :return:
        """


class Drag01(Drag):
    def __init__(self, pulse_model: Pulse) -> None:
        """

        :param pulse_model:
        """
        super().__init__(pulse_model=pulse_model)

    def drag_dep_pulse(self) -> None:
        """

        :return:
        """

        # Creat ...
        with pulse.build(backend=backend) as DRAG_inst_x:
            pulse.set_frequency(DRIVE_FREQ, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.amp_sx, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.amp_x, self.duration / 4, self._drive_beta, r'$\pi_x$'),
                       pulse.drive_channel(0))
            self.drag_inst_x = DRAG_inst_x

        # Create ...
        with pulse.build(backend=backend) as DRAG_inst_yp:
            pulse.set_frequency(DRIVE_FREQ, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.amp_sx, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.amp_x, self.duration / 4, self._drive_beta, r'$\pi_y$'),
                       pulse.drive_channel(0))
            self.drag_inst_yp = DRAG_inst_yp

        # Create ...
        with pulse.build(backend=backend) as DRAG_inst_ym:
            pulse.set_frequency(DRIVE_FREQ, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.amp_sx, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, -self.amp_x, self.duration / 4, self._drive_beta, r'$-\pi_y$'),
                       pulse.drive_channel(0))
            self.drag_inst_ym = DRAG_inst_ym

    def drag_dep_create_circuit(self):
        """

        :return:
        """
        drive_beta_circuit = np.linspace(-5, 5, 33)

        drag_gate_x = Gate(r'$\pi/2_x\cdot\pi_x$', 1, [self._drive_beta])
        drag_circ = QuantumCircuit(1, 1)
        drag_circ.append(drag_gate_x, [0])
        drag_circ.measure(0, 0)
        drag_circ.add_calibration(drag_gate_x, (0,), self.drag_inst_x, [self._drive_beta])
        drag_circuit_x = [drag_circ.assign_parameters({self._drive_beta: b}, inplace=False)
                          for b in drive_beta_circuit]

        drag_gate_yp = Gate(r'$\pi/2_x\cdot\pi_y$', 1, [self._drive_beta])
        drag_circ = QuantumCircuit(1, 1)
        drag_circ.append(drag_gate_yp, [0])
        drag_circ.measure(0, 0)
        drag_circ.add_calibration(drag_gate_yp, (0,), self.drag_inst_yp, [self._drive_beta])
        drag_circuit_yp = [drag_circ.assign_parameters({self._drive_beta: b}, inplace=False)
                           for b in drive_beta_circuit]

        drag_gate_ym = Gate(r'$\pi/2_x\cdot -\pi_y$', 1, [self._drive_beta])
        drag_circ = QuantumCircuit(1, 1)
        drag_circ.append(drag_gate_ym, [0])
        drag_circ.measure(0, 0)
        drag_circ.add_calibration(drag_gate_ym, (0,), self.drag_inst_ym, [self._drive_beta])
        drag_circuit_ym = [drag_circ.assign_parameters({self._drive_beta: b}, inplace=False)
                           for b in drive_beta_circuit]

        self.drag_circuit = drag_circuit_x + drag_circuit_yp + drag_circuit_ym


class Drag12(Drag):
    def __init__(self, pulse_model: Pulse) -> None:
        """

        :param pulse_model:
        """
        self.x_01 = Gate(r'$\pi_x^{01}$', 1, [])
        self.x_12 = Gate(r'$\pi_x^{12}$', 1, [])
        super().__init__(pulse_model=pulse_model)

    def drag_dep_pulse(self) -> None:
        """

        :return:
        """
        with pulse.build(backend=backend) as DRAG_inst_x:
            pulse.set_frequency(DRIVE_FREQ, pulse.drive_channel(0))
            pulse.call(general_12_g01(theta=np.pi,
                                      phi=0,
                                      duration=self.duration,
                                      x_amp=self.x_amp))
            pulse.set_frequency(DRIVE_FREQ + ANHAR, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.x_amp / 2, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            pulse.shift_phase(1.3089969389957465, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.x_amp, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            self.drag_inst_x = DRAG_inst_x

        with pulse.build(backend=backend) as DRAG_inst_yp:
            pulse.set_frequency(DRIVE_FREQ, pulse.drive_channel(0))
            pulse.call(general_12_g01(theta=np.pi,
                                      phi=0,
                                      duration=self.duration,
                                      x_amp=self.x_amp))
            pulse.set_frequency(DRIVE_FREQ + ANHAR, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.x_amp / 2, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            pulse.shift_phase(1.3089969389957465, pulse.drive_channel(0))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.x_amp, self.duration / 4, self._drive_beta, r'$\pi_y$'),
                       pulse.drive_channel(0))
            self.drag_inst_yp = DRAG_inst_yp

        with pulse.build(backend=backend) as DRAG_inst_ym:
            pulse.set_frequency(DRIVE_FREQ, pulse.drive_channel(0))
            pulse.call(general_12_g01(theta=np.pi,
                                      phi=0,
                                      duration=self.duration,
                                      x_amp=self.x_amp))
            pulse.set_frequency(DRIVE_FREQ + ANHAR, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, self.x_amp / 2, self.duration / 4, self._drive_beta, r'$\pi/2_x$'),
                       pulse.drive_channel(0))
            pulse.shift_phase(1.3089969389957465, pulse.drive_channel(0))
            pulse.shift_phase(-np.pi / 2, pulse.drive_channel(0))
            pulse.play(pulse.Drag(self.duration, -self.x_amp, self.duration / 4, self._drive_beta, r'$-\pi_y$'),
                       pulse.drive_channel(0))
            self.drag_inst_ym = DRAG_inst_ym

    def prepare_circuit(self) -> List:
        """
        We prepare a pre_drag level circuit
        :return:
        """
        prep0 = QuantumCircuit(1, 1)
        prep0.measure(0, 0)

        prep1 = QuantumCircuit(1, 1)
        prep1.append(self.x_01, [0])
        prep1.add_calibration(self.x_01, (0,), general_12_g01(theta=np.pi,
                                                              phi=0,
                                                              duration=self.duration,
                                                              x_amp=self.x_amp),
                              [])
        prep1.measure(0, 0)

        prep2 = QuantumCircuit(1, 1)
        prep2.append(self.x_01, [0])
        prep2.append(self.x_12, [0])
        prep2.add_calibration(self.x_01, (0,), general_12_g01(theta=np.pi,
                                                              phi=0,
                                                              duration=self.duration,
                                                              x_amp=self.x_amp),
                              [])
        prep2.add_calibration(self.x_12, (0,), drag_dephase12_g12(theta=np.pi,
                                                                  phi=0,
                                                                  duration=self.duration,
                                                                  x_amp=self.x_amp),
                              [])
        prep2.measure(0, 0)
        return [prep0, prep1, prep2]

    def drag_dep_create_circuit(self) -> None:
        """

        :return:
        """
        drive_beta_circuit = np.linspace(-5, 5, 30)

        #
        drag_gate_x = Gate(r'$\pi/2_x\cdot\pi_x$', 1, [self._drive_beta])
        drag_circ = QuantumCircuit(1, 1)
        drag_circ.append(drag_gate_x, [0])
        drag_circ.measure(0, 0)
        drag_circ.add_calibration(drag_gate_x, (0,), self.drag_inst_x, [self._drive_beta])
        drag_circ_x = [drag_circ.assign_parameters({self._drive_beta: b}, inplace=False) for b in drive_beta_circuit]

        #
        drag_gate_yp = Gate(r'$\pi/2_x\cdot\pi_y$', 1, [self._drive_beta])
        drag_circ = QuantumCircuit(1, 1)
        drag_circ.append(drag_gate_yp, [0])
        drag_circ.measure(0, 0)
        drag_circ.add_calibration(drag_gate_yp, (0,), self.drag_inst_yp, [self._drive_beta])
        drag_circ_yp = [drag_circ.assign_parameters({self._drive_beta: b}, inplace=False) for b in drive_beta_circuit]

        #
        drag_gate_ym = Gate(r'$\pi/2_x\cdot -\pi_y$', 1, [self._drive_beta])
        drag_circ_ym = []
        drag_circ = QuantumCircuit(1, 1)
        drag_circ.append(drag_gate_ym, [0])
        drag_circ.measure(0, 0)
        drag_circ.add_calibration(drag_gate_ym, (0,), self.drag_inst_ym, [self._drive_beta])
        drag_circ_ym = [drag_circ.assign_parameters({self._drive_beta: b}, inplace=False) for b in drive_beta_circuit]

        self.drag_circuit = self.prepare_circuit() + drag_circ_x + drag_circ_yp + drag_circ_ym
