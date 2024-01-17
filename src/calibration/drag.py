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

"""Drag Coefficient circuit"""
import numpy as np

from qiskit.circuit import Gate, QuantumCircuit

# from src.analyzer import DataAnalysis
from src.pulse import Pulse01, Pulse12
from src.backend.backend_ibm import EffProvider
from src.pulse_creation import GateSchedule
from src.calibration.utility import _SharedAttr
from src.calibration.discriminator import DiscriminatorQutrit
from src.exceptions.pulse_exception import (
    MissingDurationPulse,
    MissingFrequencyPulse,
    MissingAmplitudePulse
)

from typing import Union, Optional


class _DRAG(_SharedAttr):
    """
    The class act as provider + regulator for Drag Leakage.
    Drag Leakage flow: set up -> create circuit -> submit job to IBM -> get the result
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12],
                 eff_provider: EffProvider, discriminator_package: DiscriminatorQutrit,
                 backend_name: str, num_shots: int) -> None:
        """

        Args:
            pulse_model:
            eff_provider:
            backend_name:
            num_shots:

        Raises:
            MissingDurationPulse
            MissingFrequencyPulse
            MissingAmplitudePulse
        """

        if pulse_model.duration == 0:
            raise MissingDurationPulse
        if pulse_model.frequency == 0:
            raise MissingFrequencyPulse
        if pulse_model.x_amp == 0:
            raise MissingAmplitudePulse

        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)

        self.drag_sweeping_range = np.linspace(-5, 5, 100)
        self.package = discriminator_package.package

        # INTERNAL DESIGN ONLY
        self._drag_gate = Gate('DRAG', 1, [])
        self._delay_gate = Gate('5mus_delay', 1, [])

    def run_monitor(self,
                    num_shots: Optional[int] = 0,
                    meas_return: str = 'single',
                    meas_level: int = 1,
                    **kwargs):
        """

        Args:
            num_shots:
            meas_return:
            meas_level:
            **kwargs:

        """
        from qiskit.tools.monitor import job_monitor
        from qiskit import execute
        from qiskit_ibm_provider import IBMJob

        self.num_shots = num_shots if num_shots != 0 else self.num_shots
        submitted_job: IBMJob = execute(experiments=self.package,
                                        backend=self.backend,
                                        shots=self.num_shots,
                                        meas_level=meas_level,
                                        meas_return=meas_return,
                                        **kwargs)
        self.submitted_job = submitted_job.job_id()
        print(self.submitted_job)
        job_monitor(submitted_job)

    def analyze(self, job_id: Optional[str]) -> float:
        ...

    def draw(self) -> None:
        """

        """
        ...


# noinspection DuplicatedCode
class DRAG01(_DRAG):

    """

    """
    def __init__(self, pulse_model: Pulse01,
                 eff_provider: EffProvider, discriminator_package: DiscriminatorQutrit,
                 backend_name: str = 'ibmq_manila', num_shots: int = 4096) -> None:
        """

        Args:
            pulse_model:
            eff_provider:
            discriminator_package:
            backend_name:
            num_shots:
        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         discriminator_package=discriminator_package,
                         backend_name=backend_name,
                         num_shots=num_shots)

    def prepare_circuit(self) -> None:
        """

        """
        for beta in self.drag_sweeping_range:
            beta_circ = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            beta_circ.append(self._delay_gate, [self.qubit])
            beta_sweep_schedule = GateSchedule.drag(
                backend=self.backend,
                pulse_model=self.pulse_model,
                beta=beta,
                qubit=self.qubit
            )
            beta_circ.append(self._drag_gate, [self.qubit])
            beta_circ.add_calibration(self._drag_gate, [self.qubit], beta_sweep_schedule)
            beta_circ.add_calibration(self._delay_gate, [self.qubit], GateSchedule.delay(
                backend=self.backend,
                qubit=self.qubit,
            ))
            beta_circ.measure(self.qubit, self.cbit)
            self.package.append(beta_circ)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        Args:
            job_id:

        Returns:

        """
        self.pulse_model: Pulse01
        # Add Drag coefficient
        drag = self.analyze(job_id)
        self.pulse_model.drag_coeff = drag


# noinspection DuplicatedCode
class DRAG12(_DRAG):
    """

    """
    def __init__(self, pulse_model: Pulse12,
                 eff_provider: EffProvider, discriminator_package: DiscriminatorQutrit,
                 backend_name: str = 'ibmq_manila', num_shots: int = 4096) -> None:
        """

        Args:
            pulse_model:
            eff_provider:
            discriminator_package:
            backend_name:
            num_shots:
        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         discriminator_package=discriminator_package,
                         backend_name=backend_name,
                         num_shots=num_shots)

    def prepare_circuit(self) -> None:
        """

        """
        # noinspection DuplicatedCode
        for beta in self.drag_sweeping_range:
            beta_circ = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            beta_circ.append(self._delay_gate, [self.qubit])
            beta_sweep_schedule = GateSchedule.drag(
                backend=self.backend,
                qubit=self.qubit,
                beta=beta,
                pulse_model=self.pulse_model
            )
            beta_circ.append(self._drag_gate, [self.qubit])
            beta_circ.add_calibration(self._drag_gate, [self.qubit], beta_sweep_schedule)
            beta_circ.add_calibration(self._delay_gate, [self.qubit], GateSchedule.delay(
                backend=self.backend,
                qubit=self.qubit,
            ))
            beta_circ.measure(self.qubit, self.cbit)
            self.package.append(beta_circ)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        Args:
            job_id:

        Returns:

        """
        self.pulse_model: Pulse12
        # Add Drag coefficient
        drag = self.analyze(job_id)
        self.pulse_model.drag_coeff = drag
