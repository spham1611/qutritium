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

""" Transmission and reflection techniques for state transition 0-1 and 1-2.
In here TR stands for transmission and reflection
"""
import os.path
import numpy as np
import pandas as pd

from qiskit import execute
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.tools.monitor import job_monitor

from src.backend.backend_ibm import EffProvider
from src.pulse import Pulse01, Pulse12
from src.analyzer import DataAnalysis
from src.constant import QUBIT_PARA
from src.utility import fit_function
from src.calibration.mutual_attr import SharedAttr
from src.pulse_creation import GateSchedule

from numpy.typing import NDArray
from typing import List, Union, Optional
from abc import ABC, abstractmethod

mhz_unit = QUBIT_PARA.MHZ.value
ghz_unit = QUBIT_PARA.GHZ.value


def set_up_freq(center_freq: float,
                freq_span: int = 40,
                freq_step: float = 0.5) -> NDArray:
    """ Set up frequency range in NDArray form

    Arg:
        default_freq: Either DEFAULT_F01 or DEFAULT_F12
    Returns:
        frequency_range: in GHz
    """
    max_freq = center_freq + freq_span * mhz_unit / 2
    min_freq = center_freq - freq_span * mhz_unit / 2

    return np.arange(min_freq / ghz_unit, max_freq / ghz_unit, freq_step * mhz_unit / ghz_unit)


class _TR(SharedAttr, ABC):
    """ The class act as an abstract class for transmission and reflection technique used in pulse model
    TR typical flow: set up pulse and gates -> submit job to IBM -> Analyze the resulted pulse
    -> return Pulse model and save job_id.
    An example of this flow::
        from qutritium.calibration.transmission_reflection import TR01, TR12
        from qutritium.pulse import Pulse01, Pulse12

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)
        ... (TR_children_classes)

    Note:
        * You should not instantiate _TR as this is am abstract class. Use TR01 and TR12 instead

    Here is list of attributes available on the abstract "_TR" class, excluding SharedAttr attr:
        * frequency: in Hz
        * freq_sweeping_range_ghz: It is freq_sweeping_range but in ghz
        # lambda_list: return lambda_list that we use in this class
        * save_data(y_values): save given signal amplitude
        * run_monitor(package): custom run on IBMQ
        * analyze(): get the frequency of pulse after calibration
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12],
                 eff_provider: EffProvider, backend_name: str,
                 num_shots: int) -> None:
        """ _TR constructor

         Notes:
            * Frequency here is in Hz
            * freq_sweeping_range_ghz: frequency range in GHz -> convert to GHz later\
            * lambda_list: empirical, modified when running multiple experiments

        Args:
            pulse_model: Either Pulse01 or Pulse12
            eff_provider: EffProvider instance
            num_shots: number of shots
            backend_name:

        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)
        # Need to be modified by the constructor of children classes
        self.default_frequency: float = 0.
        self._package: List = []
        self.freq_sweeping_range_ghz = None

        # INTERNAL DESIGN: Used for fit_function()
        self._lambda_list = [0, 0, 0, 0]

    @property
    def lambda_list(self) -> List[float]:
        return self._lambda_list

    @lambda_list.setter
    def lambda_list(self, val_list: list) -> None:
        if len(val_list) != 4:
            raise ValueError("Lambda list does not have sufficient elements")
        self._lambda_list = val_list

    @abstractmethod
    def prepare_circuit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def modify_pulse_model(self, job_id: str = None) -> None:
        raise NotImplementedError

    def save_data(self, root_path: str, y_values) -> None:
        """ Save data in csv format

        Args:
            root_path: path to save the data of the TR experiment
            y_values:
        """
        data = {
            'freq_range': self.freq_sweeping_range_ghz,
            'f_val': y_values
        }
        if not os.path.exists(root_path):
            raise ValueError(f'Invalid filepath {root_path}')
        df = pd.DataFrame(data)
        full_path = os.path.join(root_path, f'TR of {self.pulse_model.__class__.__name__}.csv')
        df.to_csv(full_path, index=False)

    def run_monitor(self,
                    num_shots: int = 0,
                    meas_return: str = 'avg',
                    meas_level: int = 1,
                    **kwargs) -> None:
        """ Custom run execute()
        Args:
            num_shots: modify if needed
            meas_level:
            meas_return:

        Returns:

        """
        self.num_shots = num_shots if num_shots != 0 else self.num_shots
        submitted_job = execute(experiments=self._package,
                                backend=self.backend,
                                shots=self.num_shots,
                                meas_level=meas_level,
                                meas_return=meas_return, **kwargs)
        self.submitted_job = submitted_job.job_id()
        print(self.submitted_job)
        job_monitor(submitted_job)

    def analyze(self, job_id: Optional[str]) -> float:
        """ Plots IQ Data given freq_range
        Args:
            job_id: may need else we use the inherent submitted_job_id

        Returns:
            freq: Frequency after calculation
        """
        if job_id is None:
            experiment = self.backend.retrieve_job(self.submitted_job)
            analyzer = DataAnalysis(experiment=experiment)
        else:
            experiment = self.backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment)

        # Analyze data -> Save csv and output frequency
        analyzer.retrieve_data(average=True)
        fit_params, _ = fit_function(self.freq_sweeping_range_ghz, analyzer.IQ_data,
                                     lambda x, c1, q_freq, c2, c3:
                                     (c1 / np.pi) * (c2 / ((x - q_freq) ** 2 + c2 ** 2)) + c3,
                                     self.lambda_list)
        self.save_data(y_values=analyzer.IQ_data, root_path='output')
        freq = fit_params[1] * QUBIT_PARA.GHZ.value
        return freq

    def draw(self) -> None:
        """ Draw the circuit using qiskit standard library
        Implemented in next updates
        """
        pass


class TR01(_TR):
    """ Used specifically for pulse01 model
    An example of this flow::
        from qutritium.calibration.transmission_reflection import TR01
        from qutritium.pulse import Pulse01

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        eff_provider = EffProvider()
        tr_01 = TR_01(eff_provider=eff_provider, pulse_model=pulse01)
        tr_01.prepare_circuit()
        tr_01.run_monitor()
        tr_01.modify_pulse_model()
        print(tr_01.pulse_model)

    Here is list of attributes available on the ''TR_01'' class:
        + prepare_circuit(): implement abstract run_circuit() from ''_TR''
        * modify_pulse_model(): implement abstract modify_pulse_model() from ''_TR''
    """

    def __init__(self, pulse_model: Pulse01,
                 eff_provider: EffProvider, backend_name: str = "ibmq_manila",
                 num_shots: int = 4096) -> None:
        """ TR_01 constructor

        Args:
            eff_provider: EffProvider instance
            pulse_model: Pulse01
            num_shots: default 4096 shots
            backend_name: default = 'ibmq_manila'

        Returns:
            * Instance of TR01
        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)
        self.lambda_list = [10, 4.9, 1, -2]
        self.default_frequency = self.backend.defaults().qubit_freq_est[self.qubit]
        self.freq_sweeping_range_ghz: NDArray = set_up_freq(center_freq=self.default_frequency)

    def prepare_circuit(self) -> None:
        """ Calibrate single qubit state with custom pulse_model
        Notes:
            * The syntax is subject change but the idea stays the same for every update
        """

        self.pulse_model: Pulse01
        sweep_gate = Gate("sweep", 1, [])

        # Sweeping
        frequencies_hz = self.freq_sweeping_range_ghz
        for freq in frequencies_hz:
            qc_sweep = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            # noinspection DuplicatedCode
            qc_sweep.append(sweep_gate, [self.qubit])
            freq_schedule = GateSchedule.freq_gaussian(
                backend=self.backend,
                frequency=freq,
                pulse_model=self.pulse_model,
                qubit=self.qubit
            )
            qc_sweep.add_calibration(sweep_gate, [self.qubit], freq_schedule)
            qc_sweep.measure(self.qubit, self.cbit)
            self._package.append(qc_sweep)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """ Only used for debugging + getting result from past job
        Args:
            job_id: string representation of submitted job
        """
        self.pulse_model: Pulse01
        # Add frequency to pulse01
        f01 = self.analyze(job_id=job_id)
        self.pulse_model.frequency = f01


class TR12(_TR):
    """ Used specifically for Pulse12

    An example of this flow::
        from qutritium.calibration.transmission_reflection import TR12
        from qutritium.pulse import Pulse12

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)

        tr_12 = TR_12(pulse_model=pulse12)
        tr_12.prepare_circuit()
        tr_12.run_monitor()
        tr_12.modify_pulse_model()
        print(tr_12.pulse_model)

    Here is the list of attributes available "TR_12: class, excluding parents classes:
        * run_circuit(): implement abstract run_circuit() from _TR
        * modify_pulse_model(): implement abstract modify_pulse_model() from _TR
    """

    def __init__(self, pulse_model: Pulse12,
                 eff_provider: EffProvider, backend_name: str = "ibmq_manila",
                 num_shots: int = 4096) -> None:
        """

        Args:
            pulse_model: Pulse12
            eff_provider:
            num_shots: default 4096 shots
            backend_name: default = 'ibmq_manila'

        """
        super().__init__(pulse_model=pulse_model,
                         eff_provider=eff_provider,
                         backend_name=backend_name,
                         num_shots=num_shots)
        self.lambda_list = [10, 5, 1.5, -2]
        self.default_frequency = self.backend.defaults().qubit_freq_est[self.qubit] \
                                 + self.backend.properties().qubits[self.qubit][3].value * ghz_unit
        self.freq_sweeping_range_ghz: NDArray = set_up_freq(center_freq=self.default_frequency)

    def prepare_circuit(self) -> None:
        """ Some logic as TR1, however we add X gate to resemble state 12
        """

        self.pulse_model: Pulse12
        sweep_gate = Gate("sweep", 1, [])

        # Sweeping
        for freq in self.freq_sweeping_range_ghz:
            qc_sweep = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            qc_sweep.x(self.qubit)
            # noinspection DuplicatedCode
            qc_sweep.append(sweep_gate, [self.qubit])
            freq_schedule = GateSchedule.freq_gaussian(
                backend=self.backend,
                frequency=freq,
                pulse_model=self.pulse_model,
                qubit=self.qubit
            )
            qc_sweep.add_calibration(sweep_gate, [self.qubit], freq_schedule)
            qc_sweep.measure(self.qubit, self.cbit)
            self._package.append(qc_sweep)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """ Only used for debugging + getting result from past job
        Args:
            job_id: string representation of submitted job

        Returns:

        """
        self.pulse_model: Pulse12
        f12 = self.analyze(job_id=job_id)
        self.pulse_model.frequency = f12
