""" Unitary gate + conventional geometric variables that are used throughout notebooks
"""
from src.constant import QUBIT_PARA
import numpy as np


def psi(init: str) -> np.ndarray:
    """psi _summary_

    _extended_summary_

    Args:
        init (str): _description_

    Returns:
        np.ndarray: 
    """
    # Make sure init is a character
    assert len(init) == 1
    if init == 'g':
        return np.array([
            1, 0, 0
        ])
    elif init == 'e':
        return np.array([
            0, 1, 0
        ])
    elif init == 'h':
        return np.array([
            0, 0, 1
        ])
    else:
        raise ValueError


def rx01(theta: float) -> np.ndarray:
    """x01 _summary_

    _extended_summary_

    Args:
        theta (float): angle in radians

    Returns:
        np.ndarray: _description_
    """
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
        [-1j * np.sin(theta / 2), np.cos(theta / 2), 0],
        [0, 0, 1]
    ])


def rx12(theta: float) -> np.ndarray:
    """x12 _summary_

    _extended_summary_

    Args:
        theta (float): angle in radians

    Returns:
        np.ndarray: _description_
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [0, -1j * np.sin(theta / 2), np.cos(theta / 2)]
    ])


def rz01(phi: float) -> np.ndarray:
    """z01 _summary_

    _extended_summary_

    Args:
        phi (float): angle in radians

    Returns:
        _type_: _description_
    """
    return np.array([
        [np.exp(-1j * phi), 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])


def rz12(phi: float) -> np.ndarray:
    """z12 _summary_

    _extended_summary_

    Args:
        phi (float): angle in radians

    Returns:
        _type_: _description_
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, np.exp(1j * phi)]
    ])


def ry01(theta: float) -> np.ndarray:
    """y01 _summary_

    _extended_summary_

    Args:
        theta (float): np.ndarray

    Returns:
        _type_: _description_
    """
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2), 0],
        [np.sin(theta / 2), np.cos(theta / 2), 0],
        [0, 0, 1]
    ])


def ry12(theta: float) -> np.ndarray:
    """y12 _summary_

    _extended_summary_

    Args:
        theta (float): angle in radians

    Returns:
        np.ndarray: _description_
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta / 2), -np.sin(theta / 2)],
        [0, np.sin(theta / 2), np.cos(theta / 2)]
    ])


# PRR 2021: Fischer
# An arbitrary unitary in the subspace (0-1) would implement

def g01(theta: float, phi: float, var_phi: float):
    """r01 _summary_

    _extended_summary_

    Args:
        theta (float): angle in radians
        phi(float): angle in radians
        var_phi (float): angle in radians

    Returns:
        _type_: _description_
    """
    return z12(varphi) @ Rz01(phi) * Rx01(theta) * Rz01(-phi)


def g12(theta: float, phi_left: float, phi_right: float, var_phi: float):
    """r12 _summary_

    _extended_summary_

    Args:
        theta (float): angle in radians
        phi_left (float): angle in radians
        phi_right (float): angle in radians
        var_phi (float): angle in radians

    Returns:
        _type_: _description_
    """
    return z01(varphi) @ Rz12(phi) * Rx12(theta) * Rz12(-phi)
