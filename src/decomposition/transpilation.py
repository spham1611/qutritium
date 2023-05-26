"""

"""
import numpy as np
import copy
from typing import List, NamedTuple, DefaultDict, Union, Optional, Any
from collections import namedtuple, defaultdict
from src.pulse import Pulse01, Pulse12
from src.pulse_creation import Pulse_Schedule, Shift_phase, Set_frequency
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
    def get_parameters(cls, su3: NDArray) -> NamedTuple:
        """

        :param su3:
        :return:
        """
        params = namedtuple('params', 'theta1 theta2 theta3 phi1 phi2 phi3 phi4 phi5 phi6')
        if abs(np.absolute(su3[2, 2]) - 1) < 1e-6:
            theta1 = phi1 = theta2 = phi2 = 0
            phi4 = np.angle(su3[2, 2])
            phi5 = np.angle(su3[1, 1])
            phi6 = np.angle(su3[0, 0])
            phi3 = np.angle(su3[1, 0]) - phi5 + np.pi / 2
            theta3 = 2 * np.arccos(np.round(np.absolute(su3[1, 1]), 6))
        elif abs(2 * np.arccos(np.round(np.absolute(su3[2, 2]), 6)) - np.pi) < 1e-6:
            theta1 = theta3 = 0
            theta2 = np.pi
            phi2 = 0
            phi6 = np.angle(su3[0, 0])
            phi4 = -phi2 + np.angle(su3[2, 1]) + np.pi / 2
            phi5 = phi2 + np.angle(su3[1, 2]) + np.pi / 2
            phi1 = phi3 = 0
        else:
            phi4 = np.angle(su3[2, 2])
            theta2 = 2 * np.arccos(np.round(np.absolute(su3[2, 2]), 6))
            phi2 = np.angle(su3[2, 1]) - phi4 + np.pi / 2
            phi1 = np.angle(-su3[2, 0]) - phi2 - phi4
            theta1 = 2 * np.arccos(np.round(np.absolute(su3[2, 1]) / np.sin(theta2 / 2), 6))
            theta3 = 2 * np.arccos(np.round(np.absolute(su3[1, 2]) / np.sin(theta2 / 2), 6))
            phi5 = np.angle(su3[1, 2]) + phi2 + np.pi / 2
            phi3 = np.angle(np.cos(theta1 / 2) * np.cos(theta2 / 2) * np.cos(theta3 / 2)
                            - su3[1, 1] * np.exp(-1j * phi5)) + phi1
            phi6 = np.angle(-su3[0, 2]) + phi3 + phi2

        paras = params(theta1, theta2, theta3, phi1, phi2, phi3, phi4, phi5, phi6)
        return paras


class SU3_matrices:
    """

    """

    def __init__(self, su3: NDArray, qutrit_index: int, n_qutrits: int) -> None:
        """

        :param su3:
        """
        # Check dimensions
        assert su3.shape[0] == 3
        assert su3.shape[1] == 3
        self.su3: NDArray = su3
        self.qutrit_index = qutrit_index
        self.n_qutrits = n_qutrits
        self.parameters: NamedTuple = Parameter.get_parameters(su3=self.su3)

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
                                                           n_qutrit=self.n_qutrits,
                                                           parameter=[getattr(self.parameters, 'theta1'),
                                                                      getattr(self.parameters, 'phi1')]),
                                               Instruction(gate_type='g12', first_qutrit_set=self.qutrit_index,
                                                           n_qutrit=self.n_qutrits,
                                                           parameter=[getattr(self.parameters, 'theta2'),
                                                                      getattr(self.parameters, 'phi2')]),
                                               Instruction(gate_type='g01', first_qutrit_set=self.qutrit_index,
                                                           n_qutrit=self.n_qutrits,
                                                           parameter=[getattr(self.parameters, 'theta3'),
                                                                      getattr(self.parameters,
                                                                              'phi3')])]]

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
                 native_gates: Optional[List[str]]) -> None:
        """
        :param qc:
        """
        self.qc = qc
        self.n_qutrit = qc.n_qutrit
        self.ins_list = qc.operation_set
        self._su3_dictionary: DefaultDict[str, Any] = defaultdict()
        self.pulse01 = pulse01
        self.pulse12 = pulse12
        self.native_gates = native_gates if native_gates else ['u_d', 'rx', 'ry', 'rz']
        self.pulse_wrapper = []
        self.qiskit_schedule = None
        self.accumulated_phase = [np.array([0.0, 0.0])]

    def decompose(self) -> None:
        """
        Convert to SU3_matrices for further decomposition
        :return:
        """
        operation_set = self.ins_list
        n_qutrit = self.qc.n_qutrit

        for instruction in operation_set:
            if (ins_type := instruction.type()) not in self.native_gates:
                decomposed_res = SU3_matrices(instruction.gate_matrix,
                                              qutrit_index=0, n_qutrits=n_qutrit).native_list()
                self._su3_dictionary[ins_type] = decomposed_res[1]
                self.accumulated_phase.append(self.accumulated_phase[-1] + decomposed_res[0])
            else:
                self._su3_dictionary[ins_type] = [instruction]

    def convert_to_pulse_model(self):
        """
        Convert to l
        :return:
        """
        cnt = 0
        for instruction in self.ins_list:
            ins_type = instruction.type()
            phase_ud = self.accumulated_phase[cnt]
            for ins in self._su3_dictionary[ins_type]:
                gate_type = ins.type()
                if gate_type[0:2] == "rx":
                    if gate_type[2:4] == "01":
                        pulse = copy.deepcopy(self.pulse01)
                        pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                        phase_operator = Shift_phase(value=phase_ud[0],
                                                     channel=instruction.first_qutrit, subspace="01")
                    elif gate_type[2:4] == "12":
                        pulse = copy.deepcopy(self.pulse12)
                        pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                        phase_operator = Shift_phase(value=phase_ud[1],
                                                     channel=instruction.first_qutrit, subspace="12")
                    else:
                        raise Exception("The gate can not be decomposed to pulse")
                elif gate_type[0:2] == "rz":
                    if gate_type[2:4] == "01":
                        pulse = "01"
                        phase_operator = Shift_phase(value=phase_ud[0],
                                                     channel=instruction.first_qutrit, subspace="01")
                    elif gate_type[2:4] == "12":
                        pulse = "12"
                        phase_operator = Shift_phase(value=phase_ud[1],
                                                     channel=instruction.first_qutrit, subspace="12")
                    else:
                        raise Exception("The gate can not be decomposed to pulse")
                elif gate_type == 'g01':
                    pulse = copy.deepcopy(self.pulse01)
                    pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                    phase_operator = Shift_phase(value=phase_ud[0] + ins.parameter[1],
                                                 channel=instruction.first_qutrit, subspace="01")
                elif gate_type == 'g12':
                    pulse = copy.deepcopy(self.pulse12)
                    pulse.x_amp = (pulse.x_amp / np.pi) * ins.parameter[0]
                    phase_operator = Shift_phase(value=phase_ud[1] + ins.parameter[1],
                                                 channel=instruction.first_qutrit, subspace="12")
                else:
                    raise Exception("The gate can not be decomposed to pulse")
                self.pulse_wrapper.append([pulse, phase_operator, instruction.first_qutrit])
            cnt += 1

    def pulse_model_to_qiskit(self):
        """
        :return:
        """
        schedule = ScheduleBlock()
        for pul in self.pulse_wrapper:
            if type(pul[0]) in [Pulse01, Pulse12]:
                tmp_pulse = Pulse_Schedule.single_pulse_gaussian_schedule(pulse_model=pul[0], channel=pul[2])
                schedule += pul[1].generate_qiskit_phase_offset(gate_pulse=tmp_pulse)
            else:
                if pul[1].subspace == "01":
                    freq_op = Set_frequency(value=self.pulse01.frequency, channel=pul[2])
                elif pul[1].subspace == "12":
                    freq_op = Set_frequency(value=self.pulse12.frequency, channel=pul[2])
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
        :return:
        """
        cnt = 1
        for ins in self.ins_list:
            print("Phase accumulated: " + str(self.accumulated_phase[cnt]))
            for i in self._su3_dictionary[ins.type()]:
                i.print()
            cnt += 1

    def print_decompose_pulse(self):
        """
        Check the pulses after decomposition
        :return:
        """
        for pul in self.pulse_wrapper:
            print(pul)

    def print_qiskit_schedule(self):
        """
        Check the ins after decomposition
        :return:
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
