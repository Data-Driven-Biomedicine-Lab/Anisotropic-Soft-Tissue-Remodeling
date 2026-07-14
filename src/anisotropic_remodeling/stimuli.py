"""Mechanical stimuli used by the minimal remodeling law."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def principal_stretches(deformation_gradient: ArrayLike) -> FloatArray:
    """Return principal stretches in ascending order."""
    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("The deformation gradient must have shape (..., 2, 2).")
    if np.any(np.linalg.det(deformation) <= 0.0):
        raise ValueError("The deformation gradient must satisfy det(F) > 0.")

    right_cauchy_green = np.einsum("...ki,...kj->...ij", deformation, deformation)
    eigenvalues = np.linalg.eigvalsh(right_cauchy_green)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return np.sqrt(eigenvalues)


def principal_stretch_direction(deformation_gradient: ArrayLike) -> FloatArray:
    """Return the reference-direction eigenvector of the largest principal stretch."""
    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("The deformation gradient must have shape (..., 2, 2).")
    if np.any(np.linalg.det(deformation) <= 0.0):
        raise ValueError("The deformation gradient must satisfy det(F) > 0.")

    right_cauchy_green = np.einsum("...ki,...kj->...ij", deformation, deformation)
    _, eigenvectors = np.linalg.eigh(right_cauchy_green)
    direction = eigenvectors[..., :, -1]

    # Canonicalize sign for deterministic output; a and -a are physically equivalent.
    sign = np.where(
        np.abs(direction[..., 0]) >= np.abs(direction[..., 1]),
        np.sign(direction[..., 0]),
        np.sign(direction[..., 1]),
    )
    sign = np.where(sign == 0.0, 1.0, sign)
    return direction * sign[..., None]


def directional_stretch_stimulus(deformation_gradient: ArrayLike) -> FloatArray:
    """Return the magnitude of directional stretch anisotropy.

    S = |log(lambda_max) - log(lambda_min)| is objective, non-negative, and
    vanishes for isotropic stretches.
    """
    stretches = principal_stretches(deformation_gradient)
    if np.any(stretches <= 0.0):
        raise ValueError("Principal stretches must be strictly positive.")
    return np.abs(np.log(stretches[..., -1]) - np.log(stretches[..., 0]))


def hill_activation(
    stimulus: ArrayLike,
    *,
    half_saturation: float = 0.2,
    hill_exponent: float = 2.0,
) -> FloatArray:
    """Return a bounded Hill activation in [0, 1]."""
    stimulus_array = np.asarray(stimulus, dtype=float)
    if not np.all(np.isfinite(stimulus_array)) or np.any(stimulus_array < 0.0):
        raise ValueError("The remodeling stimulus must be finite and non-negative.")
    if half_saturation <= 0.0 or hill_exponent <= 0.0:
        raise ValueError("half_saturation and hill_exponent must be positive.")

    numerator = stimulus_array**hill_exponent
    denominator = half_saturation**hill_exponent + numerator
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(stimulus_array, dtype=float),
        where=denominator > 0.0,
    )


def equilibrium_structural_order(
    stimulus: ArrayLike,
    *,
    beta_min: float = 0.1,
    beta_max: float = 1.0,
    half_saturation: float = 0.2,
    hill_exponent: float = 2.0,
) -> FloatArray:
    """Map a non-negative stimulus to an equilibrium order parameter.

    A Hill law provides a bounded, monotone response. At zero stimulus the
    equilibrium value is ``beta_min``; at large stimulus it approaches
    ``beta_max``.
    """
    if not 0.0 <= beta_min <= beta_max <= 1.0:
        raise ValueError("Require 0 <= beta_min <= beta_max <= 1.")
    activation = hill_activation(
        stimulus,
        half_saturation=half_saturation,
        hill_exponent=hill_exponent,
    )
    return beta_min + (beta_max - beta_min) * activation
