"""
Support function for constructing VM
"""
import numpy as np
from numpy.linalg import inv
from numpy.linalg import LinAlgError

pi = np.pi
state_0 = [[1], [0], [0]]
state_1 = [[0], [1], [0]]
state_2 = [[0], [0], [1]]


def single_matrix_form(gate_type: str, parameter: float = None, omega=np.exp(1j * 2 * pi / 3)):
    """
    :param gate_type: quantum gate type as define in gate_set
    :param parameter: parameter of rotation gate (if needed)
    :param omega: omega for phase gate
    :return: matrix form of the qutrit gate
    """
    if gate_type == 'x01':
        return np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'rx01':
        return np.array([[np.cos(parameter / 2), -1j * np.sin(parameter / 2), 0],
                         [-1j * np.sin(parameter / 2), np.cos(parameter / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'G01':
        return np.array([[np.cos(parameter[0] / 2), -1j * np.sin(parameter[0] / 2) * np.exp(-1j * parameter[1]), 0],
                         [-1j * np.sin(parameter[0] / 2) * np.exp(1j * parameter[1]), np.cos(parameter[0] / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'x12':
        return np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]], dtype=complex)
    elif gate_type == 'rx12':
        return np.array([[1, 0, 0],
                         [0, np.cos(parameter / 2), -1j * np.sin(parameter / 2)],
                         [0, -1j * np.sin(parameter / 2), np.cos(parameter / 2)]], dtype=complex)
    elif gate_type == 'G12':
        return np.array([[1, 0, 0],
                         [0, np.cos(parameter[0] / 2), -1j * np.sin(parameter[0] / 2) * np.exp(-1j * parameter[1])],
                         [0, -1j * np.sin(parameter[0] / 2) * np.exp(1j * parameter[1]), np.cos(parameter[0] / 2)]],
                        dtype=complex)
    elif gate_type == 'I':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'x+':
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]], dtype=complex)
    elif gate_type == 'x-':
        return np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]], dtype=complex)
    elif gate_type == 'z01':
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'rz01':
        return np.array([[np.exp(-1j * parameter / 2), 0, 0],
                         [0, np.exp(1j * parameter / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'z12':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]], dtype=complex)
    elif gate_type == 'rz12':
        return np.array([[1, 0, 0],
                         [0, np.exp(-1j * parameter / 2), 0],
                         [0, 0, np.exp(1j * parameter / 2)]], dtype=complex)
    elif gate_type == 'y01':
        return np.array([[0, -1j, 0],
                         [1j, 0, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'ry01':
        return np.array([[np.cos(parameter / 2), -np.sin(parameter / 2), 0],
                         [np.sin(parameter / 2), np.cos(parameter / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'y12':
        return np.array([[1, 0, 0],
                         [0, 0, -1j],
                         [0, 1j, 0]], dtype=complex)
    elif gate_type == 'ry12':
        return np.array([[1, 0, 0],
                         [0, np.cos(parameter / 2), -np.sin(parameter / 2)],
                         [0, np.sin(parameter / 2), np.cos(parameter / 2)]], dtype=complex)
    elif gate_type == 'WH':
        return (1 / np.sqrt(3)) * np.array([[1, 1, 1],
                                            [1, omega, np.conj(omega)],
                                            [1, np.conj(omega), omega]], dtype=complex)
    elif gate_type == 'S':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, omega]], dtype=complex)
    elif gate_type == 'T':
        return np.array([[1, 0, 0],
                         [0, np.power(omega, 1 / 3), 0],
                         [0, 0, np.power(omega, -1 / 3)]], dtype=complex)
    else:
        raise Exception("This gate is not implemented yet.")


def multi_matrix_form(gate_type: str, first_index: int, second_index: int):
    """
    :param gate_type: quantum gate type as define in gate_set
    :param first_index: acting qubit
    :param second_index: control qubit
    :return: matrix form of the control-qutrit gate
    """
    if gate_type == 'CNOT':
        if second_index == first_index:
            raise Exception("Control qutrit and acting qutrit can not be the same")
        else:
            space = np.abs(first_index - second_index) - 1
            if space == 0:
                spacing = 1
            else:
                spacing = np.eye(3 ** space)
            if second_index < first_index:
                matrix = np.kron(np.kron(state_0 @ np.transpose(state_0), spacing), np.eye(3)) + \
                         np.kron(np.kron(state_1 @ np.transpose(state_1), spacing),
                                 single_matrix_form('x01') @ single_matrix_form('x12')) + \
                         np.kron(np.kron(state_2 @ np.transpose(state_2), spacing),
                                 single_matrix_form('x12') @ single_matrix_form('x01'))
            else:
                matrix = np.kron(np.kron(np.eye(3), spacing), state_0 @ np.transpose(state_0)) + \
                         np.kron(np.kron(single_matrix_form('x01') @ single_matrix_form('x12'), spacing),
                                 state_1 @ np.transpose(state_1)) + \
                         np.kron(np.kron(single_matrix_form('x12') @ single_matrix_form('x01'), spacing),
                                 state_2 @ np.transpose(state_2))
            return np.array(matrix, dtype=complex)


def statevector_to_state(state: np.array, n_qutrit: int):
    """
    :param state: State vector of qutrits
    :param n_qutrit: number of qutrit
    :return: state coefficient and state in ket form
    """
    if state.shape != (3 ** n_qutrit, 1):
        raise Exception("The dimension of the state is not align with given qubits")
    state_basis = []
    state_coeff = []
    for i in range(3 ** n_qutrit):
        if abs(complex(state[i][0])) != 0.0:
            state_basis.append(i)
            state_coeff.append(state[i][0])
    state_construction = []
    for j in range(len(state_basis)):
        sta = ''
        tmp = state_basis[j]
        for k in range(n_qutrit):
            sta += str(int(tmp % 3))
            tmp = tmp / 3
        state_construction.append(sta)
    return state_coeff, state_construction


def print_statevector(state: np.array, n_qutrit: int):
    """
    :param state: State vector of qutrits
    :param n_qutrit: number of qutrit
    Print the state vector of qutrits to ket form
    """
    state_coeff, state_cons = statevector_to_state(state, n_qutrit)
    print("State: ")
    for i in range(len(state_cons)):
        if i == len(state_cons) - 1:
            print(str(state_coeff[i]) + " |" + str(state_cons[i]) + ">")
        else:
            print(str(state_coeff[i]) + " |" + str(state_cons[i]) + "> + ")


def checking_unitary(U: np.array):
    """
    :param U: Matrix to check for unitary
    :return: True if matrix is unitary, False otherwise
    """
    try:
        prod = U @ inv(U)
    except LinAlgError:
        print("The matrix can not inverse")
        return False
    if np.absolute(np.sum(prod - np.eye(3))) < 1e-5:
        return True
    else:
        return False


def decomposition(U: np.array):
    """
    :param U: Unitary matrix to decompose into gates
    :return: parameters of the gates decomposition
    """
    if checking_unitary(U):
        if abs(np.absolute(U[2][2]) - 1) < 1e-6:
            theta_1 = phi_1 = theta_2 = phi_2 = 0
            phi_4 = np.angle(U[2][2])
            phi_5 = np.angle(U[1][1])
            phi_6 = np.angle(U[0][0])
            # phi_3 = phi_6 - pi/2 - np.angle(U[0, 1])
            phi_3 = np.angle(U[1][0]) - phi_5 + pi / 2
            theta_3 = 2 * np.arccos(np.absolute(U[1][1]))
        else:
            phi_4 = np.angle(U[2][2])
            theta_2 = 2 * np.arccos(np.round(np.absolute(U[2][2]), 6))
            phi_2 = np.angle(U[2][1]) - phi_4 + pi / 2
            phi_1 = np.angle(-1 * U[2][0]) - phi_2 - phi_4
            theta_1 = 2 * np.arccos(np.round(np.absolute(U[2][1]) / np.sin(theta_2 / 2), 6))
            theta_3 = 2 * np.arccos(np.round(np.absolute(U[1][2]) / np.sin(theta_2 / 2), 6))
            phi_5 = np.angle(U[1][2]) + phi_2 + pi / 2
            phi_3 = np.angle(
                np.cos(theta_1 / 2) * np.cos(theta_2 / 2) * np.cos(theta_3 / 2) - U[1][1] * np.exp(-1j * phi_5)) + phi_1
            phi_6 = np.angle(-1 * U[0][2]) + phi_3 + phi_2
        return theta_1, theta_2, theta_3, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6
    else:
        print("The given matrix is not unitary")
        return None
