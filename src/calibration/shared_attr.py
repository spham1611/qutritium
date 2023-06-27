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

"""This abstract class is meant to refactor attr appearing in calibration techniques"""
from abc import ABC, abstractmethod

from src.backend.backend_ibm import EffProvider
from src.pulse import Pulse01, Pulse12
from src.constant import QubitParameters

from typing import Union, Optional


class _SharedAttr(ABC):
    """
    Notes:
        * This is an abstract class and should not be instantiated
    Here is a list of shared attributes utilizing in multiple techniques:
        * pulse_model: Either Pulse01 or Pulse12
        * eff_provider: EffProvider instance
        * backend: IBMBackend which retrieves from eff_provider.backend()
        * backend_params: backend properties with the following format:
            ===============   ===============   =============
            effective_qubit   drive_frequency   anharmonicity
            int               float             float
            ===============   ===============   =============
        * qubit: effective qubit from eff_provider
        * cbit: classical qubit
        * num_shots: for execute function
        * submitted_job: job id in string format
    """
    def __init__(self, pulse_model: Union[Pulse01, Pulse12],
                 eff_provider: EffProvider, backend_name: str,
                 num_shots: int) -> None:
        """

        Args:
            pulse_model: Either Pulse01 or Pulse12
            eff_provider:
            num_shots: number of shots running in execute function
            backend_name:

        Returns:
            SharedAttr instance (do not use!)
        """
        self.pulse_model: Union[Pulse01, Pulse12] = pulse_model
        self.eff_provider: EffProvider = eff_provider
        self.backend, self.backend_params = self.eff_provider.retrieve_backend_info(backend_name)
        self.qubit: int = self.backend_params['effective_qubit']
        self.cbit: int = QubitParameters.CBIT.value
        self.num_shots: int = num_shots
        self.submitted_job: str = ''

    def prepare_circuit(self) -> None:
        pass

    def modify_pulse_model(self, job_id: str = None) -> None:
        pass

    @abstractmethod
    def run_monitor(self,
                    num_shots: Optional[int],
                    meas_return: str,
                    meas_level: int,
                    **kwargs) -> None:
        raise NotImplementedError