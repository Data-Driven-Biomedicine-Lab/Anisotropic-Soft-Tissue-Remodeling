"""Prescribed homogeneous loading paths for reproducible examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def smooth_ramp(time: float, ramp_duration: float) -> float:
    """C1-smooth ramp from zero to one using a half cosine."""
    if ramp_duration <= 0.0:
        raise ValueError("ramp_duration must be strictly positive.")
    normalized_time = np.clip(time / ramp_duration, 0.0, 1.0)
    return float(0.5 * (1.0 - np.cos(np.pi * normalized_time)))


def area_preserving_uniaxial_deformation(
    time: float,
    *,
    maximum_stretch: float,
    ramp_duration: float,
) -> FloatArray:
    """Return F = diag(lambda, 1/lambda) with a smooth loading ramp."""
    if maximum_stretch < 1.0 or not np.isfinite(maximum_stretch):
        raise ValueError("maximum_stretch must be finite and at least one.")
    load_fraction = smooth_ramp(time, ramp_duration)
    stretch = 1.0 + (maximum_stretch - 1.0) * load_fraction
    return np.array([[stretch, 0.0], [0.0, 1.0 / stretch]], dtype=float)
