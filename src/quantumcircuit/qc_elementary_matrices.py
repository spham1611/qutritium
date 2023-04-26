"""

"""
import numpy as np


def z01(phi: float) -> np.ndarray:
    """

    :param phi:
    :return:
    """
    return np.array([[np.exp(1j * phi), 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])


def z12(phi: float) -> np.ndarray:
    """

    :param phi:
    :return:
    """
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, np.exp(1j * phi)]])


def x01(theta: float = 0.) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2), 0],
                     [0, 0, 1]])


def x12(theta: float = 0.) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [0, -1j * np.sin(theta / 2), np.cos(theta / 2)]])


def y01(theta: float) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2), 0],
                     [np.sin(theta / 2), np.cos(theta / 2), 0],
                     [0, 0, 1]])


def y12(theta: float) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta / 2), -np.sin(theta / 2)],
                     [0, np.sin(theta / 2), np.cos(theta / 2)]])


def rx01(theta: float) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
        [-1j * np.sin(theta / 2), np.cos(theta / 2), 0],
        [0, 0, 1]
    ])


def rx12(theta: float) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [0, -1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])


def rz01(phi: float) -> np.ndarray:
    """

    :param phi:
    :return:
    """
    return np.array([
        [np.exp(-1j * phi), 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])


def rz12(phi: float) -> np.ndarray:
    """

    :param phi:
    :return:
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, np.exp(1j * phi)]
    ])


def ry01(theta: float) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2), 0],
        [np.sin(theta / 2), np.cos(theta / 2), 0],
        [0, 0, 1]
    ])


def ry12(theta: float) -> np.ndarray:
    """

    :param theta:
    :return:
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta / 2), -np.sin(theta / 2)],
        [0, np.sin(theta / 2), np.cos(theta / 2)]
    ])


def hdm(omega: float) -> np.ndarray:
    """
    Hadamard Gate
    :param omega:
    :return:
    """
    return 1/np.sqrt(3) * np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ])


def sdg(omega: float) -> np.ndarray:
    """

    :param omega:
    :return:
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, omega]
    ])


def identity_matrix() -> np.ndarray:
    """

    :return:
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
