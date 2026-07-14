"""Utilities for two-dimensional, head-tail symmetric fiber orientations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=float)


def normalize_vectors(vectors: ArrayLike, *, atol: float = 1e-14) -> FloatArray:
    """Normalize vectors along the last axis.

    Parameters
    ----------
    vectors:
        Array with last dimension equal to two.
    atol:
        Norms below this value are rejected.
    """
    vectors_array = _as_float_array(vectors)
    if vectors_array.shape[-1] != 2:
        raise ValueError("The last dimension of a 2D orientation vector must be 2.")

    norms = np.linalg.norm(vectors_array, axis=-1, keepdims=True)
    if np.any(norms <= atol):
        raise ValueError("Cannot normalize a zero or near-zero orientation vector.")
    return vectors_array / norms


def angle_to_vector(angle: ArrayLike) -> FloatArray:
    """Convert orientation angle(s), in radians, to unit vectors."""
    angle_array = _as_float_array(angle)
    return np.stack((np.cos(angle_array), np.sin(angle_array)), axis=-1)


def vector_to_angle(vector: ArrayLike) -> FloatArray:
    """Return orientation angle(s) in the canonical interval [0, pi)."""
    unit_vector = normalize_vectors(vector)
    angle = np.arctan2(unit_vector[..., 1], unit_vector[..., 0])
    return np.mod(angle, np.pi)


def nematic_angle_difference(angle_from: ArrayLike, angle_to: ArrayLike) -> FloatArray:
    """Smallest signed angle from one undirected orientation to another.

    The result lies in [-pi/2, pi/2), because a fiber direction is equivalent
    to the same direction rotated by pi.
    """
    angle_from_array = _as_float_array(angle_from)
    angle_to_array = _as_float_array(angle_to)
    raw_difference = angle_to_array - angle_from_array
    return 0.5 * np.arctan2(
        np.sin(2.0 * raw_difference),
        np.cos(2.0 * raw_difference),
    )


def orientation_tensor(vector: ArrayLike) -> FloatArray:
    """Return the second-order structural tensor A = a \u2297 a."""
    unit_vector = normalize_vectors(vector)
    return np.einsum("...i,...j->...ij", unit_vector, unit_vector)
