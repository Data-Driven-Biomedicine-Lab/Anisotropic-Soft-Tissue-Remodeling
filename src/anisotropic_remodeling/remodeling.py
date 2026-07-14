"""Stable update laws for fiber orientation and structural order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .orientation import angle_to_vector, nematic_angle_difference, vector_to_angle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RemodelingParameters:
    """Kinetic parameters for the minimal remodeling model."""

    orientation_rate: float = 0.25
    order_rate: float = 0.15
    beta_min: float = 0.1
    beta_max: float = 1.0
    half_saturation: float = 0.2
    hill_exponent: float = 2.0

    def __post_init__(self) -> None:
        if self.orientation_rate < 0.0 or not np.isfinite(self.orientation_rate):
            raise ValueError("orientation_rate must be finite and non-negative.")
        if self.order_rate < 0.0 or not np.isfinite(self.order_rate):
            raise ValueError("order_rate must be finite and non-negative.")
        if not 0.0 <= self.beta_min <= self.beta_max <= 1.0:
            raise ValueError("Require 0 <= beta_min <= beta_max <= 1.")
        if self.half_saturation <= 0.0 or self.hill_exponent <= 0.0:
            raise ValueError("half_saturation and hill_exponent must be positive.")


def update_fiber_orientation(
    fiber_direction: ArrayLike,
    target_direction: ArrayLike,
    *,
    rate: ArrayLike,
    dt: float,
) -> FloatArray:
    """Relax an undirected fiber orientation toward a target orientation.

    The update is exact for linear relaxation of the nematic angle over one
    time step with a fixed target. It cannot overshoot for non-negative rates.
    """
    rate_array = np.asarray(rate, dtype=float)
    if np.any(rate_array < 0.0) or not np.all(np.isfinite(rate_array)):
        raise ValueError("The orientation rate must be finite and non-negative.")
    if dt < 0.0 or not np.isfinite(dt):
        raise ValueError("The time step must be finite and non-negative.")

    current_angle = vector_to_angle(fiber_direction)
    target_angle = vector_to_angle(target_direction)
    angular_difference = nematic_angle_difference(current_angle, target_angle)
    relaxation_fraction = -np.expm1(-rate_array * dt)
    new_angle = current_angle + relaxation_fraction * angular_difference
    return angle_to_vector(new_angle)


def update_structural_order(
    structural_order: ArrayLike,
    equilibrium_order: ArrayLike,
    *,
    rate: ArrayLike,
    dt: float,
) -> FloatArray:
    """Advance beta_dot = rate * (beta_eq - beta) exactly over one step."""
    rate_array = np.asarray(rate, dtype=float)
    if np.any(rate_array < 0.0) or not np.all(np.isfinite(rate_array)):
        raise ValueError("The order rate must be finite and non-negative.")
    if dt < 0.0 or not np.isfinite(dt):
        raise ValueError("The time step must be finite and non-negative.")

    beta = np.asarray(structural_order, dtype=float)
    beta_equilibrium = np.asarray(equilibrium_order, dtype=float)
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(beta_equilibrium)):
        raise ValueError("Structural-order values must be finite.")
    if np.any((beta < 0.0) | (beta > 1.0)):
        raise ValueError("The current structural-order parameter must lie in [0, 1].")
    if np.any((beta_equilibrium < 0.0) | (beta_equilibrium > 1.0)):
        raise ValueError("The equilibrium structural-order parameter must lie in [0, 1].")

    decay = np.exp(-rate_array * dt)
    updated = beta_equilibrium + (beta - beta_equilibrium) * decay
    return np.clip(updated, 0.0, 1.0)
