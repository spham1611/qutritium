""""""
from typing import List, NamedTuple, DefaultDict, Union
from collections import namedtuple, defaultdict
from src.quantumcircuit.qc_elementary_matrices import u_d, r01, r12
from src.quantumcircuit.instruction_structure import Instruction
from src.quantumcircuit.QC import Qutrit_circuit
import numpy as np


class Parameter:
    """
    Static class
    """

    @classmethod
    def get_parameters(cls, su3: np.ndarray) -> NamedTuple:
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

    def __init__(self, su3: np.ndarray, qutrit_index: int, n_qutrits: int) -> None:
        """

        :param su3:
        """
        # Check dimensions
        assert su3.shape[0] == 3
        assert su3.shape[1] == 3
        self.su3: np.ndarray = su3
        self.qutrit_index = qutrit_index
        self.n_qutrits = n_qutrits
        self.parameters: NamedTuple = Parameter.get_parameters(su3=self.su3)

    def unitary_diagonal(self) -> np.ndarray:
        """

        :return:
        """
        return u_d(phi_1=getattr(self.parameters, 'phi6'),
                   phi_2=getattr(self.parameters, 'phi5'),
                   phi_3=getattr(self.parameters, 'phi4'))

    def rotation_theta3_01(self) -> np.ndarray:
        """

        :return:
        """
        return r01(getattr(self.parameters, 'phi3'), getattr(self.parameters, 'theta3'))

    def rotation_theta1_01(self) -> np.ndarray:
        """

        :return:
        """
        return r01(getattr(self.parameters, 'phi1'), getattr(self.parameters, 'theta1'))

    def rotation_theta2_12(self) -> np.ndarray:
        """

        :return:
        """
        return r12(getattr(self.parameters, 'phi2'), getattr(self.parameters, 'theta2'))

    def reconstruct(self) -> np.ndarray:
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
        return f"U_diagonal = {self.unitary_diagonal()}\n" \
               f"R_theta1 = {self.rotation_theta1_01()}\n" \
               f"R_theta2 = {self.rotation_theta2_12()}\n" \
               f"R_theta3 = {self.rotation_theta3_01()}\n"

    def __repr__(self) -> str:
        """

        :return:
        """
        return f"U_diagonal = {self.unitary_diagonal()}\n" \
               f"R_theta1 = {self.rotation_theta1_01()}\n" \
               f"R_theta2 = {self.rotation_theta2_12()}\n" \
               f"R_theta3 = {self.rotation_theta3_01()}\n"


class Pulse_Wrapper:
    """

    """

    def __init__(self, qc: Qutrit_circuit,
                 native_gates: List[str]) -> None:
        """
        :param qc:
        """
        self.qc = qc
        self.ins_list = qc.operation_set
        self._su3_dictionary: DefaultDict = defaultdict()
        self.native_gates = native_gates
        self.accumulated_phase = np.array([0.0, 0.0])

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
                self.accumulated_phase = self.accumulated_phase + decomposed_res[0]
                self._su3_dictionary[ins_type] = decomposed_res[1]
            else:
                self._su3_dictionary[ins_type] = instruction

    def print_decompose_ins(self):
        """
        Check the ins after decomposition
        :return:
        """
        print("Phase accumulated: " + str(self.accumulated_phase))
        for ins in self.ins_list:
            if type(self._su3_dictionary[ins.type()]) == list:
                for i in self._su3_dictionary[ins.type()]:
                    i.print()
            else:
                ins.print()

    def convert_to_pulse_model(self):
        """
        Convert to l
        :return:
        """

    def pulse_model_to_qiskit(self):
        """

        :return:
        """
        pass

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return ""
