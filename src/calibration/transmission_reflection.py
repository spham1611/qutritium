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

"""
Transmission and reflection techniques for 0-1 and 1-2
In here TR stands for transmission and reflection
"""
import numpy as np
import pandas as pd
from qiskit import execute
from qiskit.circuit import Parameter, Gate, QuantumCircuit
from qiskit.tools.monitor import job_monitor
from typing import List, Union, Tuple, Optional
from src.calibration import (
    backend,
    provider,
    QUBIT_VAL,
    DEFAULT_F01,
    DEFAULT_F12
)
from src.pulse import Pulse01, Pulse12
from src.analyzer import DataAnalysis
from src.constant import QUBIT_PARA
from src.pulse_creation import Gate_Schedule
from src.utility import fit_function
from abc import ABC, abstractmethod
from numpy import linspace
from numpy.typing import NDArray


def set_up_freq(default_freq: float) -> Tuple[NDArray, NDArray]:
    """

    Arg:
        default_freq: Either DEFAULT_F01 or DEFAULT_F12
    Returns:
        frequency_range:
        frequency_range_in_ghz:
    """
    mhz_unit = QUBIT_PARA.MHZ.value
    max_freq, min_freq = default_freq + 36 * mhz_unit, default_freq - 36 * mhz_unit
    frequency_range = linspace(min_freq, max_freq, 100)
    frequency_range_ghz = frequency_range / QUBIT_PARA.GHZ.value
    return frequency_range, frequency_range_ghz


class _TR(ABC):
    """
    The class act as an abstract class for transmission and reflection technique used in pulse model
    TR Process flow: set up pulse and gates -> submit job to IBM -> Analyze the resulted pulse
    -> return Pulse model and save job_id. The run() method does not return or take any args; it will modify
    class attribute
    An example of this flow::
        from qutritium.calibration.transmission_reflection import TR01, TR12
        from qutritium.pulse import Pulse01, Pulse12

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)

        tr_01 = TR_01(pulse_model=pulse01)
        tr_01.run()
        tr_12 = TR_12(pulse_model=pulse12)
        tr_12.run()

    Note:
        * You should not instantiate _TR as this is am abstract class. Use TR01 and TR12 instead

    Here is list of attributes available on the abstract ''_TR'' class:
        * pulse_model: either Pulse01 or Pulse12
        * num_shots: number of shots running on IBM quantum computer
        * frequency: in Hz
        * submitted_job_id: job id of job submitted on IBMQ
        * freq_sweeping_range: Sweeping frequency range for analysis
        * freq_sweeping_range_ghz: It is freq_sweeping_range but in ghz
        * qc: Our QuantumCircuit used for TR technique
        # lambda_list: return lambda_list that we use in this class
        * run(): sequential process as described above
        * save_data(y_values): save given signal amplitude
        * run_monitor(package): submit job to IBMQ
        * analyze(): plot freq_range
    """

    def __init__(self, pulse_model: Union[Pulse01, Pulse12], num_shots: int) -> None:
        """ _TR constructor

        Args:
            pulse_model: Either Pulse01 or Pulse12
            num_shots: number of shots
        """
        self.pulse_model = pulse_model
        self.num_shots = num_shots
        self.frequency: float = 0.
        self.submitted_job_id: str = ''
        self.default_freq = DEFAULT_F01 if isinstance(pulse_model, Pulse01) else DEFAULT_F12
        self.freq_sweeping_range, self.freq_sweeping_range_ghz = set_up_freq(default_freq=self.default_freq)
        self.qc: QuantumCircuit = QuantumCircuit(QUBIT_VAL + 1, 1)

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

    def run(self) -> None:
        """
        Run standard TR protocol
        """
        self.run_circuit()
        self.modify_pulse_model()
        print("Process run successfully!")

    @abstractmethod
    def run_circuit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def modify_pulse_model(self, job_id: str = None) -> None:
        raise NotImplementedError

    def save_data(self, y_values) -> None:
        """ Save data in csv format

        Args:
            y_values:
        """
        data = {
            'freq_range': self.freq_sweeping_range_ghz,
            'f_val': y_values
        }
        df = pd.DataFrame(data)
        df.to_csv(f'TR of {self.pulse_model.__class__.__name__}.csv', index=False)

    def run_monitor(self, package: List) -> None:
        """
        Submit job to IBMQ
        Args:
            package: assign parameters of quantum circuit

        Returns:

        """
        submitted_job = execute(package, backend, meas_level=1, meas_return='avg', shots=self.num_shots)
        self.submitted_job_id = submitted_job.job_id()
        job_monitor(submitted_job)

    def analyze(self, job_id: Optional[str]) -> float:
        """
        Plot IQ Data given freq_range
        Args:
            job_id: may need else we use the inherent submitted_job_id

        Returns:
            freq: Frequency after calculation
        """
        if job_id is None:
            experiment = provider.backend.retrieve_job(self.submitted_job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)
        else:
            experiment = provider.backend.retrieve_job(job_id)
            analyzer = DataAnalysis(experiment=experiment, num_shots=self.num_shots)

        # Analyze data -> Save csv and output frequency
        analyzer.retrieve_data(average=True)
        fit_params, _ = fit_function(self.freq_sweeping_range_ghz, analyzer.IQ_data,
                                     lambda x, c1, q_freq, c2, c3:
                                     (c1 / np.pi) * (c2 / ((x - q_freq) ** 2 + c2 ** 2)) + c3,
                                     self.lambda_list)
        self.save_data(y_values=analyzer.IQ_data)
        freq = fit_params[1] * QUBIT_PARA.GHZ.value
        return freq

    def draw(self) -> None:
        """
        Draw the circuit using qiskit standard library
        """
        self.qc.draw()


class TR_01(_TR):
    """ Used specifically for pulse01 model
    An example of this flow::
        from qutritium.calibration.transmission_reflection import TR01
        from qutritium.pulse import Pulse01

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        tr_01 = TR_01(pulse01)
        tr_01.run()

    Here is list of attributes available on the ''TR_01'' class:
        * frequency: Parameter - pseudo para
        * run(): inherit from ''_TR''
        + run_circuit(): implement abstract run_circuit() from ''_TR''
        * modify_pulse_model(): implement abstract modify_pulse_model() from ''_TR''
    """

    def __init__(self, pulse_model: Pulse01, num_shots: int = 20000) -> None:
        """ TR_01 constructor

        Args:
            pulse_model: Pulse01
            num_shots: default 20000 shots
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = [10, 4.9, 1, -2]
        self.frequency = Parameter('transition_freq_01')

    def run(self) -> None:
        """Can be overwritten"""
        super().run()

    def run_circuit(self) -> None:
        """
        Append an arbitrary gate, and then using the calibration -> we yield frequency range
        """
        self.pulse_model: Pulse01
        freq01_probe = Gate('Unitary', 1, [self.frequency])
        self.qc.append(freq01_probe, [QUBIT_VAL])
        self.qc.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        self.qc.add_calibration(freq01_probe, [QUBIT_VAL],
                                Gate_Schedule.single_gate_schedule(drive_freq=self.frequency,
                                                                   drive_duration=self.pulse_model.duration,
                                                                   drive_amp=self.pulse_model.x_amp),
                                [self.frequency])

        # Get the circuits from assigned frequencies
        package = [self.qc.assign_parameters({self.frequency: f}, inplace=False)
                   for f in self.freq_sweeping_range]
        self.run_monitor(package)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """
        Only used for debugging + getting result from past job
        Args:
            job_id: string representation of submitted job
        """
        self.pulse_model: Pulse01
        # Add frequency to pulse01
        f01 = self.analyze(job_id=job_id)
        self.pulse_model.frequency = f01


class TR_12(_TR):
    """ Used specifically for Pulse12

    An example of this flow::
        from qutritium.calibration.transmission_reflection import TR12
        from qutritium.pulse import Pulse12

        pulse01 = Pulse01(duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)

        tr_12 = TR_12(pulse_model=pulse12)
        tr_12.run()
    Here is the list of attributes available ''TR_12'' class:
        * run_circuit(): implement abstract run_circuit() from _TR
        * modify_pulse_model(): implement abstract modify_pulse_model() from _TR
    """

    def __init__(self, pulse_model: Pulse12, num_shots: int = 20000) -> None:
        """
        In this process, we create a new pulse12 -> Users don't need to create pulse12 from pulse12 class as there
        might be conflicts in pulse01 and 12 parameters
        Args:
            pulse_model: Pulse12
            num_shots: default 20000 shots
        """
        super().__init__(pulse_model=pulse_model, num_shots=num_shots)
        self.lambda_list = [-10, 4.8, 1, -2]
        self.frequency = Parameter('transition_freq_12')
        self.pulse01_schedule = Gate_Schedule.single_gate_schedule(
            drive_freq=self.pulse_model.pulse01.frequency,
            drive_duration=self.pulse_model.pulse01.duration,
            drive_amp=self.pulse_model.pulse01.x_amp,
        )

    def run(self) -> None:
        super().run()

    def run_circuit(self) -> None:
        """
        Some logic as TR1, however we add two gates and two calibration processes
        """
        self.pulse_model: Pulse12
        freq12_probe = Gate('Unit', 1, [self.frequency])
        x01_pi_gate = Gate(r'$X^{01}_\pi$', 1, [])
        self.qc.append(x01_pi_gate, [QUBIT_VAL])
        self.qc.append(freq12_probe, [QUBIT_VAL])
        self.qc.measure(QUBIT_VAL, QUBIT_PARA.CBIT.value)
        self.qc.add_calibration(x01_pi_gate, [QUBIT_VAL], self.pulse01_schedule)
        self.qc.add_calibration(freq12_probe, [QUBIT_VAL],
                                Gate_Schedule.single_gate_schedule(drive_freq=self.frequency,
                                                                   drive_duration=self.pulse_model.pulse01.duration,
                                                                   drive_amp=self.pulse_model.pulse01.x_amp),
                                [self.frequency])
        package = [self.qc.assign_parameters({self.frequency: f}, inplace=False)
                   for f in self.freq_sweeping_range]
        self.run_monitor(package)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """
        Only used for debugging + getting result from past job
        Args:
            job_id: string representation of submitted job

        Returns:

        """
        self.pulse_model: Pulse12
        f12 = self.analyze(job_id=job_id)
        self.pulse_model.frequency = f12
