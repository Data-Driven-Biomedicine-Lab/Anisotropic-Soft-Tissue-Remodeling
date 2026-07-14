"""Couple reconstructed polarimetric structure to spatial remodeling.

The current implementation uses prescribed compatible kinematics and evolves
only pixels accepted by an explicit validity mask. Invalid pixels are never
silently interpolated: floating-point output fields are reported as NaN there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .material import MaterialParameters, cauchy_stress, strain_energy_density
from .orientation import normalize_vectors, vector_to_angle
from .remodeling import RemodelingParameters, update_fiber_orientation, update_structural_order
from .spatial import compatible_shear_extension_deformation
from .stimuli import (
    directional_stretch_stimulus,
    equilibrium_structural_order,
    hill_activation,
    principal_stretch_direction,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PolarimetryRemodelingConfig:
    """Time and prescribed-loading parameters for image-initialized remodeling."""

    total_time: float = 20.0
    dt: float = 0.1
    ramp_duration: float = 5.0
    maximum_stretch: float = 1.18
    maximum_shear: float = 0.35
    half_height: float = 1.0

    def __post_init__(self) -> None:
        if self.total_time <= 0.0 or not np.isfinite(self.total_time):
            raise ValueError("total_time must be finite and strictly positive.")
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be finite and strictly positive.")
        if self.ramp_duration <= 0.0 or self.ramp_duration > self.total_time:
            raise ValueError("ramp_duration must lie in (0, total_time].")
        if self.maximum_stretch < 1.0 or not np.isfinite(self.maximum_stretch):
            raise ValueError("maximum_stretch must be finite and at least one.")
        if not np.isfinite(self.maximum_shear):
            raise ValueError("maximum_shear must be finite.")
        if self.half_height <= 0.0 or not np.isfinite(self.half_height):
            raise ValueError("half_height must be finite and strictly positive.")


@dataclass(frozen=True, slots=True)
class PolarimetryInitializedResult:
    """Snapshots and valid-pixel histories from polarimetry-initialized remodeling."""

    x: FloatArray
    y: FloatArray
    valid_mask: BoolArray
    time: FloatArray
    snapshot_time: FloatArray
    deformation_gradient: FloatArray
    fiber_direction: FloatArray
    fiber_angle_deg: FloatArray
    structural_order: FloatArray
    equilibrium_order: FloatArray
    stimulus: FloatArray
    strain_energy: FloatArray
    cauchy_stress: FloatArray
    mean_structural_order: FloatArray
    mean_equilibrium_order: FloatArray
    mean_stimulus: FloatArray
    mean_strain_energy: FloatArray
    mean_cauchy_stress_xx: FloatArray
    orientation_coherence: FloatArray
    mean_orientation_angle_deg: FloatArray
    mean_target_alignment: FloatArray


def _snapshot_indices(time: FloatArray, requested_times: ArrayLike) -> NDArray[np.int64]:
    requested = np.asarray(requested_times, dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("snapshot_times must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(requested)):
        raise ValueError("snapshot_times must be finite.")
    if np.any((requested < time[0]) | (requested > time[-1])):
        raise ValueError("snapshot_times must lie within the simulated time interval.")
    indices = np.array([int(np.argmin(np.abs(time - value))) for value in requested], dtype=int)
    return np.unique(indices)


def _validate_initial_fields(
    x: ArrayLike,
    y: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    valid_mask: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    fiber_array = np.asarray(fiber_direction, dtype=float)
    beta_array = np.asarray(structural_order, dtype=float)
    mask_array = np.asarray(valid_mask, dtype=bool)

    if x_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError("x and y must be two-dimensional coordinate maps.")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have identical shapes.")
    if fiber_array.shape != x_array.shape + (2,):
        raise ValueError("fiber_direction must have shape x.shape + (2,).")
    if beta_array.shape != x_array.shape:
        raise ValueError("structural_order must match the coordinate-map shape.")
    if mask_array.shape != x_array.shape:
        raise ValueError("valid_mask must match the coordinate-map shape.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("Coordinate maps must be finite.")
    if not np.any(mask_array):
        raise ValueError("valid_mask must contain at least one valid pixel.")
    if not np.all(np.isfinite(fiber_array[mask_array])):
        raise ValueError("Valid fiber directions must be finite.")
    if not np.all(np.isfinite(beta_array[mask_array])):
        raise ValueError("Valid structural-order values must be finite.")
    if np.any((beta_array[mask_array] < 0.0) | (beta_array[mask_array] > 1.0)):
        raise ValueError("Valid structural-order values must lie in [0, 1].")

    normalized_fiber = np.full_like(fiber_array, np.nan, dtype=float)
    normalized_fiber[mask_array] = normalize_vectors(fiber_array[mask_array])
    beta_clean = np.full_like(beta_array, np.nan, dtype=float)
    beta_clean[mask_array] = beta_array[mask_array]
    return x_array, y_array, normalized_fiber, beta_clean, mask_array


def _scatter_valid(
    values: FloatArray,
    valid_mask: BoolArray,
    trailing_shape: tuple[int, ...] = (),
) -> FloatArray:
    output = np.full(valid_mask.shape + trailing_shape, np.nan, dtype=float)
    output[valid_mask] = values
    return output


def _nematic_statistics_valid(fiber: FloatArray) -> tuple[float, float]:
    angle = vector_to_angle(fiber)
    mean_cosine = float(np.mean(np.cos(2.0 * angle)))
    mean_sine = float(np.mean(np.sin(2.0 * angle)))
    mean_angle = float(np.mod(0.5 * np.arctan2(mean_sine, mean_cosine), np.pi))
    coherence = float(np.hypot(mean_cosine, mean_sine))
    return mean_angle, coherence


def run_polarimetry_initialized_remodeling(
    x: ArrayLike,
    y: ArrayLike,
    initial_fiber_direction: ArrayLike,
    initial_structural_order: ArrayLike,
    valid_mask: ArrayLike,
    config: PolarimetryRemodelingConfig = PolarimetryRemodelingConfig(),
    material: MaterialParameters = MaterialParameters(),
    remodeling: RemodelingParameters = RemodelingParameters(),
    *,
    snapshot_times: ArrayLike | None = None,
) -> PolarimetryInitializedResult:
    """Evolve image-derived structure under prescribed compatible kinematics.

    Each valid image pixel is treated as an uncoupled material point. This is
    appropriate for the current prescribed-kinematics demonstrator but is not a
    substitute for a finite-element equilibrium solve. Invalid pixels remain
    excluded throughout the simulation and are stored as NaN in output fields.
    """
    x_array, y_array, fiber_field, beta_field, mask = _validate_initial_fields(
        x,
        y,
        initial_fiber_direction,
        initial_structural_order,
        valid_mask,
    )
    fiber = fiber_field[mask].copy()
    beta = beta_field[mask].copy()

    number_of_intervals = int(np.ceil(config.total_time / config.dt))
    time = np.linspace(0.0, config.total_time, number_of_intervals + 1, dtype=float)
    number_of_steps = time.size
    if snapshot_times is None:
        snapshot_times = (0.0, config.ramp_duration, config.total_time)
    selected_indices = _snapshot_indices(time, snapshot_times)
    selected_set = set(selected_indices.tolist())

    mean_beta = np.empty(number_of_steps, dtype=float)
    mean_beta_equilibrium = np.empty(number_of_steps, dtype=float)
    mean_stimulus = np.empty(number_of_steps, dtype=float)
    mean_energy = np.empty(number_of_steps, dtype=float)
    mean_sigma_xx = np.empty(number_of_steps, dtype=float)
    coherence = np.empty(number_of_steps, dtype=float)
    mean_angle_deg = np.empty(number_of_steps, dtype=float)
    mean_alignment = np.empty(number_of_steps, dtype=float)

    deformation_snapshots: list[FloatArray] = []
    fiber_snapshots: list[FloatArray] = []
    angle_snapshots: list[FloatArray] = []
    beta_snapshots: list[FloatArray] = []
    beta_equilibrium_snapshots: list[FloatArray] = []
    stimulus_snapshots: list[FloatArray] = []
    energy_snapshots: list[FloatArray] = []
    stress_snapshots: list[FloatArray] = []

    for step, current_time in enumerate(time):
        deformation_full = compatible_shear_extension_deformation(
            y_array,
            current_time,
            half_height=config.half_height,
            maximum_stretch=config.maximum_stretch,
            maximum_shear=config.maximum_shear,
            ramp_duration=config.ramp_duration,
        )
        deformation = deformation_full[mask]
        stimulus = directional_stretch_stimulus(deformation)
        beta_equilibrium = equilibrium_structural_order(
            stimulus,
            beta_min=remodeling.beta_min,
            beta_max=remodeling.beta_max,
            half_saturation=remodeling.half_saturation,
            hill_exponent=remodeling.hill_exponent,
        )
        target = principal_stretch_direction(deformation)
        energy = strain_energy_density(deformation, fiber, beta, material)
        stress = cauchy_stress(deformation, fiber, beta, material)

        nematic_angle, nematic_coherence = _nematic_statistics_valid(fiber)
        alignment = np.einsum("...i,...i->...", fiber, target) ** 2

        mean_beta[step] = float(np.mean(beta))
        mean_beta_equilibrium[step] = float(np.mean(beta_equilibrium))
        mean_stimulus[step] = float(np.mean(stimulus))
        mean_energy[step] = float(np.mean(energy))
        mean_sigma_xx[step] = float(np.mean(stress[..., 0, 0]))
        coherence[step] = nematic_coherence
        mean_angle_deg[step] = np.rad2deg(nematic_angle)
        mean_alignment[step] = float(np.mean(alignment))

        if step in selected_set:
            deformation_snapshots.append(_scatter_valid(deformation, mask, (2, 2)))
            fiber_snapshots.append(_scatter_valid(fiber, mask, (2,)))
            angle_snapshots.append(_scatter_valid(np.rad2deg(vector_to_angle(fiber)), mask))
            beta_snapshots.append(_scatter_valid(beta, mask))
            beta_equilibrium_snapshots.append(_scatter_valid(beta_equilibrium, mask))
            stimulus_snapshots.append(_scatter_valid(stimulus, mask))
            energy_snapshots.append(_scatter_valid(energy, mask))
            stress_snapshots.append(_scatter_valid(stress, mask, (2, 2)))

        if step + 1 < number_of_steps:
            step_size = float(time[step + 1] - current_time)
            activation = hill_activation(
                stimulus,
                half_saturation=remodeling.half_saturation,
                hill_exponent=remodeling.hill_exponent,
            )
            fiber = update_fiber_orientation(
                fiber,
                target,
                rate=remodeling.orientation_rate * activation,
                dt=step_size,
            )
            beta = update_structural_order(
                beta,
                beta_equilibrium,
                rate=remodeling.order_rate,
                dt=step_size,
            )

    return PolarimetryInitializedResult(
        x=x_array,
        y=y_array,
        valid_mask=mask,
        time=time,
        snapshot_time=time[selected_indices],
        deformation_gradient=np.stack(deformation_snapshots),
        fiber_direction=np.stack(fiber_snapshots),
        fiber_angle_deg=np.stack(angle_snapshots),
        structural_order=np.stack(beta_snapshots),
        equilibrium_order=np.stack(beta_equilibrium_snapshots),
        stimulus=np.stack(stimulus_snapshots),
        strain_energy=np.stack(energy_snapshots),
        cauchy_stress=np.stack(stress_snapshots),
        mean_structural_order=mean_beta,
        mean_equilibrium_order=mean_beta_equilibrium,
        mean_stimulus=mean_stimulus,
        mean_strain_energy=mean_energy,
        mean_cauchy_stress_xx=mean_sigma_xx,
        orientation_coherence=coherence,
        mean_orientation_angle_deg=mean_angle_deg,
        mean_target_alignment=mean_alignment,
    )
