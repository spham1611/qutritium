# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Density-matrix visualization: cityscape, Hinton, and ideal rho vs reconstructed rho."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_VALID_PLOT: tuple = ("city", "hinton")


def _validate_square(rho: NDArray) -> None:
    """Validate the density matrix is square."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"Invalid density matrix; got shape {rho.shape}")


def _bar3d(ax, part: NDArray, label: str) -> None:
    """3D bar chart of one real matrix on a prepared 3D axis."""
    d = part.shape[0]
    xs, ys = (a.ravel() for a in np.meshgrid(range(d), range(d), indexing="ij"))
    ax.bar3d(xs, ys, np.zeros_like(xs, dtype=float), 0.6, 0.6, part.ravel(), shade=True)
    ax.set_title(label)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xlabel("j")
    ax.set_ylabel("k")
    ax.set_zlim(-1.0, 1.0)


def _hinton(ax, matrix: NDArray, label: str) -> None:
    """Hinton diagram: square area ~ |value|, white +, black -."""
    from matplotlib.patches import Rectangle

    max_w = float(np.abs(matrix).max()) or 1.0
    d = matrix.shape[0]
    ax.set_facecolor("lightgray")
    ax.set_aspect("equal")
    for (i, j), w in np.ndenumerate(matrix):
        size = np.sqrt(abs(w) / max_w)
        color = "white" if w >= 0 else "black"
        ax.add_patch(
            Rectangle(
                (j - size / 2, i - size / 2),
                size,
                size,
                facecolor=color,
                edgecolor="gray",
            )
        )
    ax.set_title(label)
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(d - 0.5, -0.5)


def plot_density_matrix(
        rho: NDArray,
        style: str = "city",
        title: str = "Density matrix",
) -> Figure:
    """Plot Re(rho) and Im(rho) as a cityscape or Hinton diagram.

    Parameters
    ----------
    rho : NDArray
        Square density matrix (any dimension).
    style : str, optional
        ``"city"`` (3D bars, default) or ``"hinton"`` (2D area-coded).
    title : str, optional
        Figure.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``rho`` is not square or ``plot`` is unknown.
    """
    if style not in _VALID_PLOT:
        raise ValueError(f"style must be one of {_VALID_PLOT}; got {style!r}.")
    rho = np.asarray(rho, dtype=complex)
    _validate_square(rho)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    parts = [(rho.real, "Re(rho)"), (rho.imag, "Im(rho)")]
    for i, (part, label) in enumerate(parts, start=1):
        if style == "city":
            ax = fig.add_subplot(1, 2, i, projection="3d")
            _bar3d(ax, part, label)
        else:
            ax = fig.add_subplot(1, 2, i)
            _hinton(ax, part, label)
    fig.suptitle(title)
    return fig


def plot_tomography_comparison(
        rho_ideal: NDArray,
        rho_estimated: NDArray,
        fidelity: float | None = None,
        title: str = "State tomography",
) -> Figure:
    """Cityscape of ideal vs reconstructed rho.

    Rows are ideal / estimated, columns are Re / Im. Pass ``fidelity`` to
    annotate the suptitle.

    Parameters
    ----------
    rho_ideal, rho_estimated : NDArray
        Square density matrices.
    fidelity : float or None, optional
        State fidelity to display.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    ideal = np.asarray(rho_ideal, dtype=complex)
    est = np.asarray(rho_estimated, dtype=complex)
    _validate_square(ideal)
    _validate_square(est)
    if ideal.shape != est.shape:
        raise ValueError(f"shapes differ: {ideal.shape} vs {est.shape}.")

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    panels = [
        (ideal.real, "ideal  Re"),
        (ideal.imag, "ideal  Im"),
        (est.real, "estimated  Re"),
        (est.imag, "estimated  Im"),
    ]
    for i, (part, label) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        _bar3d(ax, part, label)
    suptitle = title if fidelity is None else f"{title}  (fidelity = {fidelity:.4f})"
    fig.suptitle(suptitle)
    return fig


__all__ = ["plot_density_matrix", "plot_tomography_comparison"]
