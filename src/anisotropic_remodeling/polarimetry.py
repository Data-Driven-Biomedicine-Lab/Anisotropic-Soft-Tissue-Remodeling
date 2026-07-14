"""Convert two-dimensional polarimetric maps into structural model fields.

The module deliberately separates synthetic optical-like quantities from mechanical
state variables. Retardance is mapped to a bounded structural-order *proxy*
through an explicit calibration; it is not treated as a universal direct
measurement of material anisotropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .orientation import angle_to_vector, orientation_tensor

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RetardanceCalibration:
    """Monotone calibration from retardance signal to structural order.

    Parameters
    ----------
    lower_retardance, upper_retardance:
        Retardance values mapped to ``beta_min`` and ``beta_max``. Values
        outside the interval are clipped.
    beta_min, beta_max:
        Bounds of the reconstructed structural-order proxy.
    exponent:
        Shape parameter of the monotone map. ``1`` gives a linear map.
    """

    lower_retardance: float
    upper_retardance: float
    beta_min: float = 0.0
    beta_max: float = 1.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.lower_retardance,
            self.upper_retardance,
            self.beta_min,
            self.beta_max,
            self.exponent,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Calibration parameters must be finite.")
        if self.upper_retardance <= self.lower_retardance:
            raise ValueError("upper_retardance must exceed lower_retardance.")
        if not 0.0 <= self.beta_min <= self.beta_max <= 1.0:
            raise ValueError("beta bounds must satisfy 0 <= beta_min <= beta_max <= 1.")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be strictly positive.")


@dataclass(frozen=True, slots=True)
class PolarimetryStructureResult:
    """Structural fields reconstructed from azimuth and retardance maps."""

    azimuth_rad: FloatArray
    fiber_direction: FloatArray
    structure_tensor: FloatArray
    retardance_order: FloatArray
    local_coherence: FloatArray
    structural_order: FloatArray
    valid_mask: BoolArray


def canonicalize_nematic_azimuth(
    azimuth_rad: ArrayLike,
    *,
    orientation_offset_rad: float = 0.0,
) -> FloatArray:
    """Return head-tail symmetric azimuths in the canonical interval [0, pi)."""
    azimuth = np.asarray(azimuth_rad, dtype=float)
    if not np.isfinite(orientation_offset_rad):
        raise ValueError("orientation_offset_rad must be finite.")
    return np.mod(azimuth + orientation_offset_rad, np.pi)


def retardance_to_order_proxy(
    retardance: ArrayLike,
    calibration: RetardanceCalibration,
) -> FloatArray:
    """Map retardance monotonically to a bounded structural-order proxy."""
    signal = np.asarray(retardance, dtype=float)
    normalized = (signal - calibration.lower_retardance) / (
        calibration.upper_retardance - calibration.lower_retardance
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    shaped = normalized**calibration.exponent
    return calibration.beta_min + (
        calibration.beta_max - calibration.beta_min
    ) * shaped


def _validate_window_size(window_size: int) -> None:
    if not isinstance(window_size, (int, np.integer)):
        raise TypeError("window_size must be an integer.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")


def _box_sum_2d(values: FloatArray, window_size: int) -> FloatArray:
    """Return constant-padded centered box sums using an integral image."""
    _validate_window_size(window_size)
    if values.ndim != 2:
        raise ValueError("Local coherence is defined for two-dimensional maps.")

    radius = window_size // 2
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    return (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )


def local_nematic_coherence(
    azimuth_rad: ArrayLike,
    *,
    window_size: int = 9,
    weights: ArrayLike | None = None,
    valid_mask: ArrayLike | None = None,
) -> FloatArray:
    r"""Compute local head-tail symmetric orientation coherence in [0, 1].

    The doubled-angle representation removes the artificial discontinuity
    between 0 and pi:

    ``q = <w exp(2 i alpha)> / <w>``, ``coherence = |q|``.
    """
    _validate_window_size(window_size)
    azimuth = np.asarray(azimuth_rad, dtype=float)
    if azimuth.ndim != 2:
        raise ValueError("azimuth_rad must be a two-dimensional map.")

    finite_azimuth = np.isfinite(azimuth)
    if valid_mask is None:
        valid = finite_azimuth
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != azimuth.shape:
            raise ValueError("valid_mask must match the azimuth map shape.")
        valid = finite_azimuth & mask

    if weights is None:
        weight = np.ones_like(azimuth)
    else:
        weight = np.asarray(weights, dtype=float)
        if weight.shape != azimuth.shape:
            raise ValueError("weights must match the azimuth map shape.")
        if np.any(np.isfinite(weight) & (weight < 0.0)):
            raise ValueError("weights must be non-negative.")

    effective_weight = np.where(valid & np.isfinite(weight), weight, 0.0)
    safe_azimuth = np.where(valid, azimuth, 0.0)

    weight_sum = _box_sum_2d(effective_weight, window_size)
    cosine_sum = _box_sum_2d(
        effective_weight * np.cos(2.0 * safe_azimuth),
        window_size,
    )
    sine_sum = _box_sum_2d(
        effective_weight * np.sin(2.0 * safe_azimuth),
        window_size,
    )

    coherence = np.full_like(azimuth, np.nan, dtype=float)
    supported = weight_sum > 0.0
    coherence[supported] = np.hypot(cosine_sum[supported], sine_sum[supported]) / weight_sum[
        supported
    ]
    coherence[supported] = np.clip(coherence[supported], 0.0, 1.0)
    return coherence


def polarimetry_to_structure(
    azimuth_rad: ArrayLike,
    retardance: ArrayLike,
    calibration: RetardanceCalibration,
    *,
    orientation_offset_rad: float = 0.0,
    minimum_valid_retardance: float | None = None,
    external_valid_mask: ArrayLike | None = None,
    coherence_window: int = 9,
    combine_with_coherence: bool = True,
) -> PolarimetryStructureResult:
    """Convert polarimetric maps into model-ready structural fields.

    Invalid pixels are represented by ``NaN`` in floating-point output fields
    and by ``False`` in ``valid_mask``. The structural-order field is either the
    retardance-derived proxy alone or its product with local nematic coherence.
    """
    azimuth_input = np.asarray(azimuth_rad, dtype=float)
    retardance_input = np.asarray(retardance, dtype=float)
    if azimuth_input.ndim != 2 or retardance_input.ndim != 2:
        raise ValueError("azimuth_rad and retardance must both be two-dimensional maps.")
    if azimuth_input.shape != retardance_input.shape:
        raise ValueError("azimuth_rad and retardance must have identical shapes.")
    if minimum_valid_retardance is not None and not np.isfinite(minimum_valid_retardance):
        raise ValueError("minimum_valid_retardance must be finite when provided.")

    valid = np.isfinite(azimuth_input) & np.isfinite(retardance_input)
    if minimum_valid_retardance is not None:
        valid &= retardance_input >= minimum_valid_retardance
    if external_valid_mask is not None:
        external = np.asarray(external_valid_mask, dtype=bool)
        if external.shape != valid.shape:
            raise ValueError("external_valid_mask must match the input map shape.")
        valid &= external

    azimuth = canonicalize_nematic_azimuth(
        azimuth_input,
        orientation_offset_rad=orientation_offset_rad,
    )
    retardance_order = retardance_to_order_proxy(retardance_input, calibration)
    coherence_weights = np.where(valid, np.maximum(retardance_order, 1.0e-12), 0.0)
    coherence = local_nematic_coherence(
        azimuth,
        window_size=coherence_window,
        weights=coherence_weights,
        valid_mask=valid,
    )

    if combine_with_coherence:
        structural_order = retardance_order * coherence
    else:
        structural_order = retardance_order.copy()

    fiber_direction = angle_to_vector(azimuth)
    structure = orientation_tensor(fiber_direction)

    azimuth = np.where(valid, azimuth, np.nan)
    fiber_direction = np.where(valid[..., None], fiber_direction, np.nan)
    structure = np.where(valid[..., None, None], structure, np.nan)
    retardance_order = np.where(valid, retardance_order, np.nan)
    coherence = np.where(valid, coherence, np.nan)
    structural_order = np.where(valid, structural_order, np.nan)

    return PolarimetryStructureResult(
        azimuth_rad=azimuth,
        fiber_direction=fiber_direction,
        structure_tensor=structure,
        retardance_order=retardance_order,
        local_coherence=coherence,
        structural_order=structural_order,
        valid_mask=valid,
    )
