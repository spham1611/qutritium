'''
Support function for constructing VM
'''
import numpy as np

pi = np.pi


def matrix_form(gate_type, parameter=None, omega=np.exp(1j * 2 * pi / 3)):
    if gate_type == 'x01':
        return np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    elif gate_type == 'x12':
        return np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    elif gate_type == 'z01':
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif gate_type == 'z12':
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif gate_type == 'y01':
        return np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 1]])
    elif gate_type == 'y12':
        return np.array([[1, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    elif gate_type == 'WH':
        return (1 / np.sqrt(3)) * np.array([[1, 1, 1], [1, omega, np.conj(omega)], [1, np.conj(omega), omega]])
    elif gate_type == 'S':
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, omega]])
    elif gate_type == 'T':
        return np.array([[1, 0, 0], [0, np.power(omega, 1 / 3), 0], [0, 0, np.power(omega, -1 / 3)]])
    else:
        raise Exception("This gate is not implemented yet.")


def statevector_to_state(state=np.array(), n_quibts=int):
    if state.shape != (3 ** n_quibts, 1):
        raise Exception("The dimension of the state is not align with given qubits")
    '''
    The final part is not finished yet
    '''
