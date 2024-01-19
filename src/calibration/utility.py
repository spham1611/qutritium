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

"""This abstract class is meant to refactor attributes appearing in calibration classes"""
from abc import ABC

from typing import Optional

from qiskit import pulse

from src.backend.backend_ibm import CustomProvider
from src.pulse import Pulse, Pulse01, Pulse12
from src.constant import QubitParameters


class _SetAttribute(ABC):
    """
    Notes:
        * This is an abstract class and should not be instantiated as it is only meant for programming convenience
    Here is a list of set attributes used in calibration functions:
        * pulse_model: Either Pulse01 or Pulse12
        * eff_provider: CustomProvider instance
        * backend: IBMBackend which retrieves from eff_provider.backend()
        * backend_params: backend properties with the following format:
            ===============   ===============   =============
            effective_qubit   drive_frequency   anharmonicity
            int               float             float
            ===============   ===============   =============
        * package: initialize package List -> deliver circuits
        * qubit: effective qubit from eff_provider
        * cbit: classical qubit
        * num_shots: for execute function
        * submitted_job: job id in string format
    """

    def __init__(self,
                 custom_provider: CustomProvider,
                 backend_name: str,
                 num_shots: int,
                 model_space: str,
                 pulse_model: Optional[Pulse],
                 pulse_connected=None,
                 drive_duration_ns: float = 120.) -> None:
        """

        Args:
            model_space:
            custom_provider:
            backend_name:
            num_shots:
            pulse_model:
            pulse_connected: User can access the related pulse via Pulse getter, and it is recommended to do so
            drive_duration_ns:
        """
        self.custom_provider: CustomProvider = custom_provider
        self.backend, self.backend_params = self.custom_provider.retrieve_backend_info(backend_name)
        self.model_space = model_space
        self.drive_duration_sec = drive_duration_ns * QubitParameters.ns.value
        self.pulse_connected = pulse_connected
        self.pulse_model = self.set_pulse_model() if not pulse_model else pulse_model

        self.qubit: int = self.backend_params['effective_qubit']
        self.cbit: int = QubitParameters.CBIT.value
        self.num_shots: int = num_shots
        self.submitted_job: str = ''

    def set_pulse_model(self) -> Pulse:
        """
        Set pulse model based ok Qiskit pulse -> to run on IBM devices. A number of functions are demonstrated in Qiskit
        tutorial that users can find here: https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html
        """
        granularity = self.backend.configuration().timing_constraints['granularity']

        def get_closest_multiple(value, base_number):
            return int(value + base_number / 2) - (int(value + base_number / 2) % base_number)

        def get_close_multiple_of_granularity(num):
            return get_closest_multiple(num, granularity)

        if self.model_space == '01':
            return Pulse01(x_amp=0.1, drive_duration=get_close_multiple_of_granularity(
                pulse.seconds_to_samples(self.drive_duration_sec)
            ), pulse12=self.pulse_connected)
        elif self.model_space == '12':
            return Pulse12(x_amp=0.1, drive_duration=get_close_multiple_of_granularity(
                pulse.seconds_to_samples(self.drive_duration_sec)
            ), pulse01=self.pulse_connected)
        else:
            raise ValueError("Invalid quantum space!. Only write either: '01' or '12'")

    def prepare_circuit(self) -> None:
        pass

    def modify_pulse_model(self, job_id: str = None) -> None:
        pass
