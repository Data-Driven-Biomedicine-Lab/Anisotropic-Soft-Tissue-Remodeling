"""Deterministic synthetic datasets for executable benchmark notebooks.

Synthetic fields provide deterministic, known-ground-truth benchmarks for
orientation and structural-order reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .polarimetry import RetardanceCalibration

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SyntheticPolarimetryBenchmark:
    """Noisy synthetic azimuth and retardance maps with known latent fields."""

    x: FloatArray
    y: FloatArray
    true_azimuth_rad: FloatArray
    true_structural_order: FloatArray
    observed_azimuth_rad: FloatArray
    observed_retardance: FloatArray
    external_valid_mask: BoolArray
    calibration: RetardanceCalibration


def synthetic_polarimetry_benchmark(
    *,
    width: float = 2.0,
    height: float = 1.0,
    number_of_points_x: int = 81,
    number_of_points_y: int = 41,
    random_seed: int = 7,
) -> SyntheticPolarimetryBenchmark:
    """Create a reproducible rectangular polarimetry benchmark.

    The latent orientation varies smoothly across the sample. Structural order
    contains a localized low-order region. Independent Gaussian noise is added
    to azimuth and retardance, and one low-signal optical defect is inserted to
    exercise validity masking.
    """
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and strictly positive.")
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("height must be finite and strictly positive.")
    if number_of_points_x < 3 or number_of_points_y < 3:
        raise ValueError("Each coordinate direction requires at least three points.")

    x_axis = np.linspace(0.0, width, number_of_points_x)
    y_axis = np.linspace(0.0, height, number_of_points_y)
    x, y = np.meshgrid(x_axis, y_axis, indexing="xy")

    true_angle_deg = (
        55.0
        - 28.0 * x / width
        + 10.0
        * np.sin(2.0 * np.pi * y / height)
        * np.sin(np.pi * x / width)
    )
    true_azimuth = np.mod(np.deg2rad(true_angle_deg), np.pi)

    low_order_region = np.exp(
        -(
            ((x - 0.625 * width) / (0.14 * width)) ** 2
            + ((y - 0.58 * height) / (0.18 * height)) ** 2
        )
    )
    true_order = 0.52 - 0.18 * low_order_region

    calibration = RetardanceCalibration(
        lower_retardance=0.20,
        upper_retardance=0.80,
        beta_min=0.10,
        beta_max=0.70,
        exponent=1.0,
    )
    normalized_order = (true_order - calibration.beta_min) / (
        calibration.beta_max - calibration.beta_min
    )
    noiseless_retardance = calibration.lower_retardance + (
        calibration.upper_retardance - calibration.lower_retardance
    ) * np.clip(normalized_order, 0.0, 1.0)

    random = np.random.default_rng(random_seed)
    observed_azimuth = true_azimuth + random.normal(
        0.0,
        np.deg2rad(2.0),
        size=x.shape,
    )
    observed_retardance = noiseless_retardance + random.normal(
        0.0,
        0.012,
        size=x.shape,
    )

    optical_defect = (
        ((x - 0.24 * width) / (0.05 * width)) ** 2
        + ((y - 0.28 * height) / (0.09 * height)) ** 2
        < 1.0
    )
    observed_retardance[optical_defect] = 0.12
    external_mask = np.ones_like(x, dtype=bool)

    return SyntheticPolarimetryBenchmark(
        x=x,
        y=y,
        true_azimuth_rad=true_azimuth,
        true_structural_order=true_order,
        observed_azimuth_rad=observed_azimuth,
        observed_retardance=observed_retardance,
        external_valid_mask=external_mask,
        calibration=calibration,
    )
