""""""
from typing import NamedTuple, DefaultDict, List, Optional
from collections import defaultdict, namedtuple
from src.quantumcircuit.qc_elementary_matrices import u_d, r01, r12
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

    def __init__(self, su3: np.ndarray) -> None:
        """

        :param su3:
        """
        # Check dimensions
        assert su3.shape[0] == 3
        assert su3.shape[1] == 3
        self.su3: np.ndarray = su3
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


class Matrix_Wrapper:
    """

    """
    def __init__(self, qc: Qutrit_circuit,
                 native_gates: Optional[List[str]]) -> None:
        """
        :param qc:
        """
        self.qc = qc
        self._su3_dictionary: DefaultDict = defaultdict()
        self.native_gates = native_gates if native_gates else ['rx01', 'rx12', 'rz01', 'rz12']

    def transpile(self) -> None:
        """

        :return:
        """
        operation_set = self.qc.operation_set
        # TODO: Define native gates and matrix form of it
        native_gate = []
        for instruction in operation_set:
            if instr_type := instruction.type() not in native_gate:
                self._su3_dictionary[instr_type] = SU3_matrices(instruction.gate_matrix).reconstruct()
            else:
                self._su3_dictionary[instr_type] = ...

    def convert_to_pulse_model(self):
        """
        Convert to l
        :return:
        """
        pass

    def pulse_model_to_qiskit(self):
        """

        :return:
        """
        pass

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return ""
