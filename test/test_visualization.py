"""Smoke tests for qutritium.visualization (matplotlib, Agg backend)."""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from qutritium.tomography.visualization import (  # noqa: E402
    plot_density_matrix,
    plot_tomography_comparison,
)

_RHO = np.diag([0.5, 0.3, 0.2]).astype(complex)


class TestPlotDensityMatrix:
    @pytest.mark.parametrize("style", ["city", "hinton"])
    def test_returns_figure(self, style):
        assert isinstance(plot_density_matrix(_RHO, style=style), Figure)

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="style"):
            plot_density_matrix(_RHO, style="bogus")

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            plot_density_matrix(np.ones((2, 3)))


class TestPlotComparison:
    def test_returns_figure(self):
        assert isinstance(plot_tomography_comparison(_RHO, _RHO, fidelity=1.0), Figure)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            plot_tomography_comparison(_RHO, np.eye(9))
