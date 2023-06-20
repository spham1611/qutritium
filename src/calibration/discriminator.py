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

""" Discriminator circuit and iq plot function """
import matplotlib.pyplot as plt

from qiskit import pulse, QuantumCircuit
from qiskit.circuit import Gate
from qiskit import execute
from qiskit_ibm_provider.job import job_monitor

from src.pulse import Pulse12
from src.backend.backend_ibm import EffProvider
from src.calibration.shared_attr import SharedAttr
from src.constant import QubitParameters
from src.analyzer import DataAnalysis

from typing import List


mhz_unit = QubitParameters.MHZ.value
ghz_unit = QubitParameters.GHZ.value


class DiscriminatorQutrit(SharedAttr):
    """ A Simple Discriminator class that set up circuit and plot the iq graph
    An example of using this class::

        from qutritium.calibration.discriminator import Discriminator
        from qutritium.backend.backend_ibm import EffProvider
        from qutritium.pulse import Pulse01, Pulse12

        efF_provider = EffProvider()
        pulse01 = Pulse01(duration=144, x_amp=0.2)
        pulse12 = Pulse12(pulse01=pulse01, duration=pulse01.duration, x_amp=pulse01.x_amp)

        ...Running Rough Rabi

        discriminator = Discriminator(eff_provider, pulse12, num_shots=4096)
        discriminator.prepare_circuit()
        discriminator.run_monitor()
        discriminator.plot_iq()

    Here is a list of available attributes of "DiscriminatorQutrit" class:
        * prepare_circuit(): prepares a discriminator with different QuantumCircuits
        * run_monitor(): run custom execute
        * plot_iq(): plot iq 012 graph
    """
    def __init__(self, eff_provider: EffProvider, pulse_model: Pulse12,
                 backend_name: str = 'ibmq_manila', num_shots=4096,
                 delay_time: int = 22496) -> None:
        """ ctor

        Args:
            eff_provider:
            pulse_model:
            backend_name:
            num_shots:
            delay_time:
        """
        super().__init__(eff_provider=eff_provider, pulse_model=pulse_model,
                         backend_name=backend_name, num_shots=num_shots)
        self.ramsey_frequency01 = self.pulse_model.pulse01.frequency + (4 - 3.2160804020100504) * mhz_unit
        self.ramsey_frequency12 = self.pulse_model.frequency + (5 - (5.2612+4.5226)/2) * mhz_unit
        self._delay_gate = Gate('5mus_delay', 1, [])
        self._x12 = Gate('X12', 1, [])

        # Delay schedule
        with pulse.build(backend=self.backend) as delay_schedule:
            drive_chan = pulse.drive_channel(self.qubit)
            pulse.delay(delay_time, drive_chan)
        self._delay_schedule = delay_schedule

        # x12_schedule
        with pulse.build(backend=self.backend) as x12_schedule:
            drive_chan = pulse.drive_channel(self.qubit)
            pulse.set_frequency(self.ramsey_frequency12, drive_chan)
            pulse.play(pulse.Gaussian(duration=self.pulse_model.duration,
                                      sigma=self.pulse_model.sigma, amp=self.pulse_model.x_amp), drive_chan)
        self._x12_schedule = x12_schedule

        # Package
        self._package: List = []

    def prepare_circuit(self) -> None:
        """ Ground state + 2 excited states """

        ground_state_prep = QuantumCircuit(self.qubit + 1, self.cbit + 1)
        ground_state_prep.append(self._delay_gate, [self.qubit])
        ground_state_prep.add_calibration(self._delay_gate, [self.qubit], self._delay_schedule)
        # noinspection DuplicatedCode
        ground_state_prep.measure(self.qubit, self.cbit)

        first_excited_state_prep = QuantumCircuit(self.qubit + 1, self.cbit + 1)
        first_excited_state_prep.append(self._delay_gate, [self.qubit])
        first_excited_state_prep.add_calibration(self._delay_gate, [self.qubit], self._delay_schedule)
        first_excited_state_prep.x(self.qubit)
        # noinspection DuplicatedCode
        first_excited_state_prep.measure(self.qubit, self.cbit)

        second_excited_state_prep = QuantumCircuit(self.qubit + 1, self.cbit + 1)
        second_excited_state_prep.append(self._delay_gate, [self.qubit])
        second_excited_state_prep.add_calibration(self._delay_gate, [self.qubit], self._delay_schedule)
        second_excited_state_prep.x(self.qubit)
        second_excited_state_prep.append(self._x12, [self.qubit])
        second_excited_state_prep.add_calibration(self._x12, [self.qubit], self._x12_schedule)
        second_excited_state_prep.measure(self.qubit, self.cbit)

        self._package = [ground_state_prep, first_excited_state_prep, second_excited_state_prep]

    def run_monitor(self,
                    num_shots: int = 4096,
                    meas_return='single',
                    meas_level=1) -> None:
        self.num_shots = num_shots if num_shots != 0 else self.num_shots
        discriminator_job = execute(experiments=self._package,
                                    backend=self.backend,
                                    shots=self.num_shots,
                                    meas_return=meas_return,
                                    meas_level=meas_level)
        self.submitted_job = discriminator_job.job_id()
        print(self.submitted_job)
        job_monitor(discriminator_job)

    def plot_iq(self,
                x_min: int = -30,
                x_max: int = 30,
                y_min: int = -30,
                y_max: int = 30) -> None:
        """ Plot iq 012 state

        Args:
            x_min:
            x_max:
            y_min:
            y_max:
        """
        discriminator_job = self.backend.retrieve_job(self.submitted_job)
        discriminator_data = DataAnalysis(discriminator_job)
        discriminator_data.retrieve_data(True)
        discriminator_data.iq_012_plot(x_min, x_max, y_min, y_max)
        plt.savefig('output/0 1 2 Discrimination.svg', format='svg', dpi=1200)
