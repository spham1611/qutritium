import numpy as np
from numpy.typing import NDArray

pi = np.pi
state_0 = [[1], [0], [0]]
state_1 = [[0], [1], [0]]
state_2 = [[0], [0], [1]]


def x_plus() -> NDArray:
    return np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]], dtype=complex)


def x_minus() -> NDArray:
    return np.array([[0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 0]], dtype=complex)


def z01() -> NDArray:
    return np.array([[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, 1]], dtype=complex)


def z12() -> NDArray:
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, -1]], dtype=complex)


def x01() -> NDArray:
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 1]], dtype=complex)


def x12() -> NDArray:
    return np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0]], dtype=complex)


def y01() -> NDArray:
    return np.array([[0, -1j, 0],
                     [1j, 0, 0],
                     [0, 0, 1]], dtype=complex)


def y12() -> NDArray:
    return np.array([[1, 0, 0],
                     [0, 0, -1j],
                     [0, 1j, 0]], dtype=complex)


def rx01(theta: float) -> NDArray:
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
        [-1j * np.sin(theta / 2), np.cos(theta / 2), 0],
        [0, 0, 1]], dtype=complex)


def rx12(theta: float) -> NDArray:
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [0, -1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def rz01(phi: float) -> NDArray:
    return np.array([
        [np.exp(-1j * phi), 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=complex)


def rz12(phi: float) -> NDArray:
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, np.exp(1j * phi)]], dtype=complex)


def ry01(theta: float) -> NDArray:
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2), 0],
        [np.sin(theta / 2), np.cos(theta / 2), 0],
        [0, 0, 1]], dtype=complex)


def ry12(theta: float) -> NDArray:
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta / 2), -np.sin(theta / 2)],
        [0, np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def hdm(omega: float = np.exp(1j * 2 * pi / 3)) -> NDArray:
    return 1 / np.sqrt(3) * np.array([
        [1, 1, 1],
        [1, omega, omega ** 2],
        [1, omega ** 2, omega]], dtype=complex)


def sdg(omega: float = np.exp(1j * 2 * pi / 3)) -> NDArray:
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, omega]], dtype=complex)


def tdg(omega: float = np.exp(1j * 2 * pi / 3)) -> NDArray:
    return np.array([
        [1, 0, 0],
        [0, np.power(omega, 1 / 3), 0],
        [0, 0, np.power(omega, -1 / 3)]], dtype=complex)


def Identity() -> NDArray:
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=complex)


def r01(phi: float, theta: float) -> NDArray:
    return rz01(-phi) @ rx01(theta) @ rz01(phi)


def r12(phi: float, theta: float) -> NDArray:
    return rz12(phi) @ rx12(theta) @ rz12(-phi)  # note that R12 is reversed


def u_d(phi_1: float, phi_2: float, phi_3: float) -> NDArray:
    return np.array([[np.exp(1j * phi_1), 0, 0],
                     [0, np.exp(1j * phi_2), 0],
                     [0, 0, np.exp(1j * phi_3)]])


def cnot(first_index: int, second_index: int):
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
                     np.kron(np.kron(state_1 @ np.transpose(state_1), spacing), x01() @ x12()) + \
                     np.kron(np.kron(state_2 @ np.transpose(state_2), spacing), x12() @ x01())
        else:
            matrix = np.kron(np.kron(np.eye(3), spacing), state_0 @ np.transpose(state_0)) + \
                     np.kron(np.kron(x01() @ x12(), spacing), state_1 @ np.transpose(state_1)) + \
                     np.kron(np.kron(x12() @ x01(), spacing), state_2 @ np.transpose(state_2))
        return np.array(matrix, dtype=complex)
