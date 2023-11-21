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

import numpy as np
import copy
from qiskit_ibm_provider import IBMBackend
from typing import List, NamedTuple, DefaultDict, Union, Optional, Any
from collections import namedtuple, defaultdict
from src.pulse import Pulse01, Pulse12
from src.pulse_creation import Shift_phase, Set_frequency, GateSchedule
from src.quantumcircuit.qc_elementary_matrices import u_d, r01, r12
from src.quantumcircuit.instruction_structure import Instruction
from src.quantumcircuit.QC import Qutrit_circuit
from qiskit.pulse.schedule import ScheduleBlock
from numpy.typing import NDArray


class Parameter:
    """
    Static class
    """
    @classmethod
    def get_parameters(cls, U: NDArray) -> NamedTuple:
        """
        Decompose su3 matrices to parameters of selected gates
        Args:
            su3: SU3 Matrix

        Returns: Parameters constructing the gates

        """
        pi = np.pi
        params = namedtuple('params', 'theta1 theta2 theta3 phi1 phi2 phi3 phi4 phi5 phi6')
        if np.round(abs(np.absolute(U[2, 2])), 6) == 1:
            if np.round(abs(np.absolute(U[0, 0])), 6) != 0:
                theta_1 = phi_1 = theta_2 = phi_2 = 0
                phi_4 = np.angle(U[2, 2])
                phi_5 = np.angle(U[1, 1])
                phi_6 = np.angle(U[0, 0])
                # phi_3 = phi_6 - pi/2 - np.angle(U[0, 1])
                phi_3 = np.angle(U[1, 0]) - phi_5 + pi / 2
                theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 1]), 6))
            else:
                theta_1 = phi_1 = theta_2 = phi_2 = phi_3 = 0
                theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 1]), 6))
                phi_4 = np.angle(U[2, 2])
                phi_6 = np.angle(U[0, 1]) + phi_3 + pi / 2
                phi_5 = np.angle(U[1, 0]) - phi_3 + pi / 2
        elif np.round(abs(np.absolute(U[2, 2])), 6) == 0:
            theta_1 = 2 * np.arccos(np.round(np.absolute(U[2, 1]), 6))
            theta_2 = pi
            theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 2]), 6))
            phi_1 = phi_2 = phi_3 = 0
            if np.round(abs(np.absolute(U[2, 0])), 6) != 0:
                phi_4 = np.angle(-U[2, 0])
                if np.round(abs(np.absolute(U[0, 2])), 6) != 0:
                    phi_5 = np.angle(-U[1, 1])
                    phi_6 = np.angle(-U[0, 2])
                else:
                    phi_5 = np.angle(U[1, 2]) + pi / 2
                    phi_6 = np.angle(U[0, 1]) + pi / 2
            if np.round(abs(np.absolute(U[1, 0])), 6) != 0:
                phi_4 = np.angle(U[2, 1]) + pi / 2
                phi_5 = np.angle(U[1, 0]) + pi / 2
                phi_6 = np.angle(-U[0, 2])
            if np.round(abs(np.absolute(U[0, 0])), 6) != 0:
                phi_4 = np.angle(U[2, 1]) + pi / 2
                phi_5 = np.angle(U[1, 2]) + pi / 2
                phi_6 = np.angle(U[1, 1])
        else:
            phi_4 = np.angle(U[2, 2])
            theta_2 = 2 * np.arccos(np.round(np.absolute(U[2, 2]), 6))
            phi_2 = np.angle(U[2, 1]) - phi_4 + pi / 2
            phi_1 = np.angle(-U[2, 0]) - phi_2 - phi_4
            theta_1 = 2 * np.arccos(np.round(np.absolute(U[2, 1]) / np.sin(theta_2 / 2), 6))
            theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 2]) / np.sin(theta_2 / 2), 6))
            phi_5 = np.angle(U[1, 2]) + phi_2 + pi / 2
            phi_3 = np.angle(np.cos(theta_1 / 2) * np.cos(theta_2 / 2) * np.cos(theta_3 / 2) - U[1, 1] * np.exp(
                -1j * phi_5)) + phi_1
            phi_6 = np.angle(-U[0, 2]) + phi_3 + phi_2
        paras = params(theta_1, theta_2, theta_3, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6)
        return paras


class SU3_matrices:

    def __init__(self, su3: NDArray, qutrit_index: int, n_qutrits: int) -> None:
        """

        Args:
            su3:
            qutrit_index:
            n_qutrits:
        """
        # Check dimensions
        assert su3.shape[0] == 3
        assert su3.shape[1] == 3
        self.su3: NDArray = su3
        self.qutrit_index = qutrit_index
        self.n_qutrits = n_qutrits
        self.parameters: NamedTuple = Parameter.get_parameters(U=self.su3)

    def unitary_diagonal(self) -> NDArray:
        """

        :return:
        """
        return u_d(phi_1=getattr(self.parameters, 'phi6'),
                   phi_2=getattr(self.parameters, 'phi5'),
                   phi_3=getattr(self.parameters, 'phi4'))

    def rotation_theta3_01(self) -> NDArray:
        """

        :return:
        """
        return r01(phi=getattr(self.parameters, 'phi3'), theta=getattr(self.parameters, 'theta3'))

    def rotation_theta1_01(self) -> NDArray:
        """

        :return:
        """
        return r01(getattr(self.parameters, 'phi1'), getattr(self.parameters, 'theta1'))

    def rotation_theta2_12(self) -> NDArray:
        """

        :return:
        """
        return r12(getattr(self.parameters, 'phi2'), getattr(self.parameters, 'theta2'))

    def reconstruct(self) -> NDArray:
        """

        :return:
        """
        return (
                self.unitary_diagonal()
                @ self.rotation_theta3_01()
                @ self.rotation_theta2_12()
                @ self.rotation_theta1_01()
        )

    def native_list(self) -> List[Union[List[float], List[Instruction]]]:
        """

        :return:
        """
        phase01 = float(getattr(self.parameters, 'phi6') - getattr(self.parameters, 'phi5'))
        phase12 = float(getattr(self.parameters, 'phi5') - getattr(self.parameters, 'phi4'))
        return [np.array([phase01, phase12]), [Instruction(gate_type='g01', first_qutrit_set=self.qutrit_index,
                                                           second_qutrit_set=None,
                                                           n_qutrit=self.n_qutrits,
                                                           parameter=[getattr(self.parameters, 'theta1'),
                                                                      getattr(self.parameters, 'phi1')]),
                                               Instruction(gate_type='g12', first_qutrit_set=self.qutrit_index,
                                                           second_qutrit_set=None,
                                                           n_qutrit=self.n_qutrits,
                                                           parameter=[getattr(self.parameters, 'theta2'),
                                                                      getattr(self.parameters, 'phi2')]),
                                               Instruction(gate_type='g01', first_qutrit_set=self.qutrit_index,
                                                           second_qutrit_set=None,
                                                           n_qutrit=self.n_qutrits,
                                                           parameter=[getattr(self.parameters, 'theta3'),
                                                                      getattr(self.parameters,
                                                                              'phi3')])]]

    def decomposed_into_qc(self):
        decomposed_qc = Qutrit_circuit(n_qutrit=self.n_qutrits, initial_state=None)
        decomposed_qc.add_gate("g01", first_qutrit_set=self.qutrit_index, parameter=[getattr(self.parameters, 'theta1'),
                                                                                     getattr(self.parameters, 'phi1')])
        decomposed_qc.add_gate("g12", first_qutrit_set=self.qutrit_index, parameter=[getattr(self.parameters, 'theta2'),
                                                                                     getattr(self.parameters, 'phi2')])
        decomposed_qc.add_gate("g01", first_qutrit_set=self.qutrit_index, parameter=[getattr(self.parameters, 'theta3'),
                                                                                     getattr(self.parameters, 'phi3')])
        decomposed_qc.add_gate("u_d", first_qutrit_set=self.qutrit_index, parameter=[getattr(self.parameters, 'phi6'),
                                                                                     getattr(self.parameters, 'phi5'),
                                                                                     getattr(self.parameters, 'phi4')])
        return decomposed_qc

    def __str__(self) -> str:
        """

        :return:
        """
        return f"U_diagonal:\n{self.unitary_diagonal()}\n" \
               f"R_theta1:\n{self.rotation_theta1_01()}\n" \
               f"R_theta2:\n{self.rotation_theta2_12()}\n" \
               f"R_theta3:\n{self.rotation_theta3_01()}\n"

    def __repr__(self) -> str:
        """

        :return:
        """
        return f"U_diagonal:\n{self.unitary_diagonal()}\n" \
               f"R_theta1:\n{self.rotation_theta1_01()}\n" \
               f"R_theta2:\n{self.rotation_theta2_12()}\n" \
               f"R_theta3:\n{self.rotation_theta3_01()}\n"


class Pulse_Wrapper:
    """

    """

    def __init__(self, pulse01: Pulse01,
                 pulse12: Pulse12,
                 /,
                 qc: Qutrit_circuit,
                 native_gates: Optional[List[str]],
                 backend: IBMBackend) -> None:
        """
        :param qc:
        """
        self.qc = qc
        self.n_qutrit = qc.n_qutrit
        self.ins_list = qc.operation_set
        self._su3_dictionary: DefaultDict[str, Any] = defaultdict()
        self.pulse01 = pulse01
        self.pulse12 = pulse12
        self.backend = backend
        self.native_gates = native_gates if native_gates else ['u_d', 'rx', 'ry', 'rz']
        self.pulse_wrapper = []
        self.qiskit_schedule = None
        self.accumulated_phase = [np.array([0.0, 0.0])]

    def decompose(self) -> None:
        """

        Convert to SU3_matrices for further decomposition

        """
        operation_set = self.ins_list
        n_qutrit = self.qc.n_qutrit

        for instruction in operation_set:
            if type(instruction) == str:
                continue
            if (ins_type := instruction.type()) not in self.native_gates:
                decomposed_res = SU3_matrices(instruction.gate_matrix,
                                              qutrit_index=0, n_qutrits=n_qutrit).native_list()
                self._su3_dictionary[ins_type] = decomposed_res[1]
                self.accumulated_phase.append(self.accumulated_phase[-1] + decomposed_res[0])
            else:
                self._su3_dictionary[ins_type] = [instruction]

    def convert_to_pulse_model(self):
        """
        Convert to internal Pulse Model
        """
        cnt = 0
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        advance_phase = np.array([0.0, 0.0])  # Implementation of phase advance
        for instruction in self.ins_list:
            if type(instruction) == str:
                continue
            ins_type = instruction.type()
            phase_ud = self.accumulated_phase[cnt]
            for ins in self._su3_dictionary[ins_type]:
                gate_type = ins.type()
                if gate_type[0:2] == "rx":
                    if gate_type[2:4] == "01":
                        pulse = copy.deepcopy(self.pulse01)
                        pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                        phase_operator = Shift_phase(value=phase_ud[0],
                                                     channel=instruction.first_qutrit, subspace="01",
                                                     backend=self.backend)
                    elif gate_type[2:4] == "12":
                        pulse = copy.deepcopy(self.pulse12)
                        pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                        phase_operator = Shift_phase(value=phase_ud[1],
                                                     channel=instruction.first_qutrit, subspace="12",
                                                     backend=self.backend)
                    else:
                        raise Exception("The gate can not be decomposed to pulse")
                elif gate_type[0:2] == "rz":
                    if gate_type[2:4] == "01":
                        pulse = "01"
                        phase_operator = Shift_phase(value=phase_ud[0],
                                                     channel=instruction.first_qutrit, subspace="01",
                                                     backend=self.backend)
                    elif gate_type[2:4] == "12":
                        pulse = "12"
                        phase_operator = Shift_phase(value=phase_ud[1],
                                                     channel=instruction.first_qutrit, subspace="12",
                                                     backend=self.backend)
                    else:
                        raise Exception("The gate can not be decomposed to pulse")
                elif gate_type == 'g01':
                    pulse = copy.deepcopy(self.pulse01)
                    pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                    phase_operator = Shift_phase(value=phase_ud[0] + ins.parameter[1] + advance_phase[0],
                                                 # Adding
                                                 # phase advance
                                                 channel=instruction.first_qutrit, subspace="01",
                                                 backend=self.backend)
                    advance_phase[1] += alpha  # Accumulate phase advance
                elif gate_type == 'g12':
                    pulse = copy.deepcopy(self.pulse12)
                    pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                    phase_operator = Shift_phase(value=phase_ud[1] + ins.parameter[1] + advance_phase[1],
                                                 # Adding
                                                 # phase advance
                                                 channel=instruction.first_qutrit, subspace="12",
                                                 backend=self.backend)
                    advance_phase[0] += beta
                    advance_phase[1] += gamma  # Accumulate phase advance
                else:
                    raise Exception("The gate can not be decomposed to pulse")
                self.pulse_wrapper.append([pulse, phase_operator, instruction.first_qutrit])
            cnt += 1

    def pulse_model_to_qiskit(self):
        """
        Returns: Convert to Qiskit Pulse Model
        """
        schedule = ScheduleBlock()
        for pul in self.pulse_wrapper:
            if type(pul[0]) in [Pulse01, Pulse12]:
                tmp_pulse = GateSchedule.x_amp_gaussian(pulse_model=pul[0], qubit=pul[2],
                                                        backend=self.backend, x_amp=pul[0].x_amp)
                schedule += pul[1].generate_qiskit_phase_offset(gate_pulse=tmp_pulse)
            else:
                if pul[1].subspace == "01":
                    freq_op = Set_frequency(value=self.pulse01.frequency, channel=pul[2], backend=self.backend)
                elif pul[1].subspace == "12":
                    freq_op = Set_frequency(value=self.pulse12.frequency, channel=pul[2], backend=self.backend)
                else:
                    raise Exception("The pulse can not be translated to Qiskit pulse")
                freq_schedule = freq_op.generate_qiskit_freq()
                phase_schedule = pul[1].generate_qiskit_phase(coeff=1.0)
                schedule = schedule + freq_schedule + phase_schedule
        self.qiskit_schedule = schedule
        return schedule

    def print_decompose_ins(self):
        """

        Check the ins after decomposition

        """
        cnt = 0
        for ins in self.ins_list:
            if type(ins) == str:
                continue
            print("Phase accumulated: " + str(self.accumulated_phase[cnt]))
            for i in self._su3_dictionary[ins.type()]:
                i.print()
            cnt += 1

    def print_decompose_pulse(self):
        """

        Check the pulses after decomposition

        """
        for pul in self.pulse_wrapper:
            print(pul)

    def print_qiskit_schedule(self):
        """

        Check the ins after decomposition

        """

        if self.qiskit_schedule is not None:
            print(self.qiskit_schedule)
            self.qiskit_schedule.draw()
        else:
            raise Exception("Required conversion to Qiskit ScheduleBlock")

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return ""