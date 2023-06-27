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

"""Fine Tune techniques"""
import numpy as np
from numpy.typing import NDArray

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from src.calibration.shared_attr import _SharedAttr
from src.backend.backend_ibm import EffProvider
from src.pulse import Pulse01, Pulse12
from src.pulse_creation import GateSchedule

from typing import Union, List, Optional


class _FineTune(_SharedAttr):
    """
    The class act as provider + regulator for Fine Tune techniques
    """

    def __int__(self, pulse_model: Union[Pulse01, Pulse12],
                eff_provider: EffProvider, backend_name: str,
                num_shots: int, alpha_lb: float,
                alpha_ub: float, data_points: int) -> None:
        """

        Args:
            pulse_model:
            eff_provider:
            backend_name:
            num_shots:
            alpha_lb:
            alpha_ub:

        Returns:

        """
        super().__init__(eff_provider=eff_provider, backend_name=backend_name,
                         pulse_model=pulse_model, num_shots=num_shots)
        self.lower_bound = self.pulse_model.x_amp - self.pulse_model.x_amp * alpha_lb
        self.upper_bound = self.pulse_model.x_amp + self.pulse_model.x_amp * alpha_ub
        self.ft_amp_range: NDArray = np.linspace(self.lower_bound, self.upper_bound, data_points)

        self.package: List[QuantumCircuit] = []
        # INTERNAL Attributes
        self._ft_gate = Gate("fine_tune", 1, [])

    def run_monitor(self,
                    num_shots: Optional[int],
                    meas_return: str = 'avg',
                    meas_level: int = 1,
                    **kwargs) -> None:
        """

        Args:
            num_shots:
            meas_return:
            meas_level:
            **kwargs:

        Returns:

        """
        pass

    def analyze(self, job_id: Optional[str] = '') -> float:
        ...

    def draw(self) -> None:
        ...


class FT01(_FineTune):
    """

    """

    def __int__(self, pulse_model: Pulse01,
                eff_provider: EffProvider, backend_name: str,
                num_shots: int = 4096, alpha_lb: float = 0.01,
                alpha_ub: float = 0.01, data_points: int = 100) -> None:
        """

        Args:
            pulse_model:
            eff_provider:
            backend_name:
            num_shots:
            alpha_lb:
            alpha_ub:
            data_points:

        Returns:

        """
        super().__int__(eff_provider=eff_provider, pulse_model=pulse_model,
                        backend_name=backend_name, num_shots=num_shots,
                        alpha_lb=alpha_lb, alpha_ub=alpha_ub,
                        data_points=data_points)

    def prepare_circuit(self) -> None:
        """
        """
        for amp in self.ft_amp_range:
            qc_ft = QuantumCircuit(self.qubit + 1, self.cbit + 1)
            ft_schedule = GateSchedule.x_amp_gaussian(
                backend=self.backend,
                pulse_model=self.pulse_model,
                x_amp=amp,
                qubit=self.qubit
            )
            for i in range(1251):
                qc_ft.append(self._ft_gate, [self.qubit])
            qc_ft.add_calibration(self._ft_gate, [self.qubit], ft_schedule)
            qc_ft.measure(self.qubit, self.cbit)
            self.package.append(qc_ft)

    def modify_pulse_model(self, job_id: str = None) -> None:
        """

        Args:
            job_id:

        """
        self.pulse_model: Pulse01
        self.pulse_model.x_amp = self.analyze(job_id)
