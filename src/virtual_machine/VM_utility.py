'''
Support function for constructing VM
'''
import numpy as np

pi = np.pi


def matrix_form(gate_type, parameter=None, omega=np.exp(1j * 2 * pi / 3)):
    """
    :param gate_type: quantum gate type as define in gate_set
    :param parameter: parameter of rotation gate (if needed)
    :param omega: omega for phase gate
    :return: matrix form of the qutrit gate
    """
    if gate_type == 'x01':
        return np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]])
    elif gate_type == 'x12':
        return np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]])
    elif gate_type == 'z01':
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])
    elif gate_type == 'z12':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]])
    elif gate_type == 'y01':
        return np.array([[0, -1j, 0],
                         [1j, 0, 0],
                         [0, 0, 1]])

    elif gate_type == 'y12':
        return np.array([[1, 0, 0],
                         [0, 0, -1j],
                         [0, 1j, 0]])
    elif gate_type == 'WH':
        return (1 / np.sqrt(3)) * np.array([[1, 1, 1],
                                            [1, omega, np.conj(omega)],
                                            [1, np.conj(omega), omega]])
    elif gate_type == 'S':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, omega]])
    elif gate_type == 'T':
        return np.array([[1, 0, 0],
                         [0, np.power(omega, 1 / 3), 0],
                         [0, 0, np.power(omega, -1 / 3)]])
    else:
        raise Exception("This gate is not implemented yet.")


def statevector_to_state(state, n_qutrit=int):
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
        if abs(state[i][0]) != 0.0:
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


def print_statevector(state, n_qutrit=int):
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
