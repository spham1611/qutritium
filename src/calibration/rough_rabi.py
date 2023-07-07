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

"""Rough rabi techniques"""
import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import QuantumCircuit, Gate

from src.backend.backend_ibm import EffProvider
from src.pulse import Pulse01, Pulse12
from src.calibration.shared_attr import _SharedAttr
from src.exceptions.pulse_exception import MissingFrequencyPulse
from src.pulse_creation import GateSchedule

from typing import Optional, List, Union


class _RoughRabi(_SharedAttr):
    """ Provides basic attributes of RoughRabi technique and inherits SharedAttr available attributes
    A typical flow would be prepare_circuit -> run_monitor -> analyze IBMJob::

        from qutritium.calibration.rough_rabi import RoughRabi01, RoughRabi12

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)

        ... (RoughRabi_children_class)

    Here is a list of available attributes of "_RoughRabi" class, excluding SharedAttr attr:
        * lambda_list: list of parameters for fit function
        * x_amp_sweeping_range: for sweeping amplitude
        * reset_sweeping_range():
        * prepare_circuit():
        * modify_pulse_model(): run analyze() and get the x_amp
        * save_data(): for saving data only
        * run_monitor():
        * analyze(): plots iq_data + get x_amp from fit function
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12],
                 eff_provider: EffProvider, backend_name: str,
                 num_shots: int) -> None:
        """ _RoughRabi Constructor

        Notes:
            * This class should not be instantiated. Use RoughRabi01 or RoughRabi12

        Args:
            pulse_model:
            eff_provider:
            num_shots:
        """
        if pulse_model.frequency == 0:
            raise MissingFrequencyPulse('Need to have frequency. Assign it from TR function or default drive frequency')
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)
        self.analyzer = None
        self._package: List = []

        # INTERNAL DESIGN ONLY
        self._lambda_list: Optional[List] = None
        self._x_amp_sweeping_range = np.linspace(-1., 1., 100)
        self._rabi_gate = Gate("Rabi", 1, [])
        self._rr_fit: Optional[NDArray] = None

    @property
    def lambda_list(self) -> List[float]:
        return self._lambda_list

    @lambda_list.setter
    def lambda_list(self, val_list: list) -> None:
        """

        Args:
            val_list:

        Returns:
            lambda_list:
        """
        if len(val_list) != 4:
            raise ValueError("Lambda list does not have sufficient elements")
        self._lambda_list = val_list

    @property
    def x_amp_sweeping_range(self) -> NDArray:
        return self._x_amp_sweeping_range

    @property
    def rr_fit(self) -> NDArray:
        return self._rr_fit

    def reset_sweeping_range(self, min_val: float, max_val: float, steps: int) -> None:
        """
        Reset the sweeping range
        Args:
            min_val:
            max_val:
            steps:
        """
        self._x_amp_sweeping_range = np.linspace(min_val, max_val, steps)

    def run_monitor(self,
                    num_shots: int = 0,
                    meas_level: int = 1,
                    meas_return: str = 'avg',
                    **kwargs) -> None:
        """
        Run custom execute()
        Args:
            num_shots:
            meas_level:
            meas_return:
            **kwargs:

        """
        from qiskit_ibm_provider.job import job_monitor
        from qiskit import execute

        self.num_shots = num_shots if num_shots != 0 else self.num_shots
        submitted_job = execute(experiments=self._package,
                                backend=self.backend,
                                meas_level=meas_level,
                                meas_return=meas_return,
                                shots=self.num_shots,
                                **kwargs)
        self.submitted_job = submitted_job.job_id()
        print(self.submitted_job)
        job_monitor(submitted_job)

    def analyze(self, job_id: str) -> float:
        """

        Args:
            job_id:

        Returns:

        """
        from src.utility import fit_function
        from src.analyzer import DataAnalysis

        if not job_id:
            experiment = self.eff_provider.retrieve_job(self.submitted_job)
        else:
            experiment = self.eff_provider.retrieve_job(job_id)
        self.analyzer = DataAnalysis(experiment)

        self.analyzer.retrieve_data(average=True)
        fit_params, self._rr_fit = fit_function(self._x_amp_sweeping_range, self.analyzer.IQ_data,
                                                lambda x, c1, c2, drive_period, phi:
                                                (c1 * np.cos(2 * np.pi * x / drive_period - phi) + c2),
                                                [5, 0, 0.5, 0])

        x_amp = (fit_params[2] / 2)
        return x_amp


class RoughRabi01(_RoughRabi):
    """ Provides rough rabi circuit and overrides methods from "_RoughRabi" class
    Here is an example of the rough rabi flow::

        from qutritium.calibration.rough_rabi import RoughRabi01
        from qutritium.backend.backend_ibm import EffProvider

        pulse01 = Pulse01(frequency=4.9e9, duration=144, x_amp=0.2)
        eff_provider = EffProvider()

        rr01 = RoughRabi01(pulse01, eff_provider)
        rr01.prepare_circuit()
        rr01.run_monitor()
        rr01.modify_pulse_model()

    Here is a list of available functions of "RoughRabi01" class, excluding inheritance attributes
        * prepare_circuit(): override "_RoughRabi"
        * modify_pulse_model(): override "_RoughRabi"
    """

    def __init__(self, pulse_model: Pulse01,
                 eff_provider, backend_name='ibmq_manila',
                 num_shots: int = 4096) -> None:
        """ Ctor

        Args:
            pulse_model:
            eff_provider
            num_shots:
        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)
        self.lambda_list = [5, 0, 0.5, 0]

    def prepare_circuit(self) -> None:
        """ runs over x_amp sweeping range
        Returns:

        """
        self.pulse_model: Pulse01

        # Sweeping
        for x_amp in self.x_amp_sweeping_range:
            qc_rabi01 = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            # noinspection DuplicatedCode
            qc_rabi01.append(self._rabi_gate, [self.qubit])
            rabi_schedule = GateSchedule.x_amp_gaussian(
                backend=self.backend,
                x_amp=x_amp,
                pulse_model=self.pulse_model,
                qubit=self.qubit
            )
            qc_rabi01.add_calibration(self._rabi_gate, [self.qubit], rabi_schedule)
            qc_rabi01.measure(self.qubit, self.cbit)
            self._package.append(qc_rabi01)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        Args:
            job_id:

        Returns:

        """
        self.pulse_model: Pulse01
        self.pulse_model.x_amp = self.analyze(job_id=job_id)


class RoughRabi12(_RoughRabi):
    """ Overrides "_RoughRabi", used for Pulse12
    Here is an example of the rough rabi flow::

        from qutritium.calibration.rough_rabi import RoughRabi01
        from qutritium.backend.backend_ibm import EffProvider

        pulse01 = Pulse01(frequency=4.9e9, duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, frequency=pulse01.frequency,
                          x_amp=pulse01.x_amp, duration=pulse01.duration)
        eff_provider = EffProvider()

        rr01 = RoughRabi01(pulse01, eff_provider)
        rr01.prepare_circuit()
        rr01.run_monitor()
        rr01.modify_pulse_model()

    Here is a list of available functions of "RoughRabi01" class, excluding inheritance attributes
        * prepare_circuit(): overrides "_RoughRabi"
        * modify_pulse_model(): overrides "_RoughRabi"
    """

    def __init__(self, pulse_model: Pulse12,
                 eff_provider: EffProvider, backend_name='ibmq_manila',
                 num_shots: int = 4096) -> None:
        """ Ctor

        Args:
            pulse_model:
            eff_provider
            num_shots:
        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)
        self.lambda_list = [5, 0, 0.4, 0]

    def prepare_circuit(self) -> None:
        """ sweeps over x_amp with given range

        Returns:

        """
        self.pulse_model: Pulse12

        for x_amp in self.x_amp_sweeping_range:
            qc_rabi12 = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            qc_rabi12.x(self.qubit)
            # noinspection DuplicatedCode
            qc_rabi12.append(self._rabi_gate, [self.qubit])
            rabi_schedule = GateSchedule.x_amp_gaussian(
                backend=self.backend,
                x_amp=x_amp,
                pulse_model=self.pulse_model,
                qubit=self.qubit
            )
            qc_rabi12.add_calibration(self._rabi_gate, [self.qubit], rabi_schedule)
            qc_rabi12.measure(self.qubit, self.cbit)
            self._package.append(qc_rabi12)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        Args:
            job_id:

        Returns:

        """
        self.pulse_model: Pulse12
        self.pulse_model.x_amp = self.analyze(job_id=job_id)
