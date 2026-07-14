"""Spatially heterogeneous prescribed-kinematics remodeling demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .loading import smooth_ramp
from .material import MaterialParameters, cauchy_stress, strain_energy_density
from .orientation import angle_to_vector, normalize_vectors, vector_to_angle
from .remodeling import RemodelingParameters, update_fiber_orientation, update_structural_order
from .stimuli import (
    directional_stretch_stimulus,
    equilibrium_structural_order,
    hill_activation,
    principal_stretch_direction,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SpatialSimulationConfig:
    """Configuration for a rectangular spatial-field remodeling example."""

    nx: int = 61
    ny: int = 41
    half_width: float = 1.5
    half_height: float = 1.0
    total_time: float = 20.0
    dt: float = 0.1
    ramp_duration: float = 5.0
    maximum_stretch: float = 1.18
    maximum_shear: float = 0.35
    mean_fiber_angle_deg: float = 45.0
    angle_amplitude_deg: float = 25.0
    beta_background: float = 0.35
    beta_defect_depth: float = 0.22
    defect_center_x: float = 0.30
    defect_center_y: float = -0.15
    defect_width_x: float = 0.35
    defect_width_y: float = 0.25

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must both be at least three.")
        if self.half_width <= 0.0 or self.half_height <= 0.0:
            raise ValueError("Domain half-widths must be strictly positive.")
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
        if not np.isfinite(self.mean_fiber_angle_deg):
            raise ValueError("mean_fiber_angle_deg must be finite.")
        if self.angle_amplitude_deg < 0.0 or not np.isfinite(self.angle_amplitude_deg):
            raise ValueError("angle_amplitude_deg must be finite and non-negative.")
        if not 0.0 <= self.beta_background <= 1.0:
            raise ValueError("beta_background must lie in [0, 1].")
        if self.beta_defect_depth < 0.0 or not np.isfinite(self.beta_defect_depth):
            raise ValueError("beta_defect_depth must be finite and non-negative.")
        if self.beta_background - self.beta_defect_depth < 0.0:
            raise ValueError("The prescribed beta defect would make beta negative.")
        if self.defect_width_x <= 0.0 or self.defect_width_y <= 0.0:
            raise ValueError("Defect widths must be strictly positive.")


@dataclass(frozen=True, slots=True)
class SpatialSimulationResult:
    """Selected field snapshots and global histories from a spatial simulation."""

    x: FloatArray
    y: FloatArray
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


def rectangular_grid(config: SpatialSimulationConfig) -> tuple[FloatArray, FloatArray]:
    """Return a Cartesian mesh over the reference rectangle."""
    x_coordinates = np.linspace(-config.half_width, config.half_width, config.nx)
    y_coordinates = np.linspace(-config.half_height, config.half_height, config.ny)
    return np.meshgrid(x_coordinates, y_coordinates, indexing="xy")


def synthetic_fiber_field(
    x: ArrayLike,
    y: ArrayLike,
    *,
    half_width: float,
    half_height: float,
    mean_angle_deg: float,
    angle_amplitude_deg: float,
) -> FloatArray:
    """Create a smooth synthetic nematic orientation field."""
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("Domain half-widths must be strictly positive.")
    if angle_amplitude_deg < 0.0:
        raise ValueError("angle_amplitude_deg must be non-negative.")

    angle_deg = mean_angle_deg + angle_amplitude_deg * (
        np.sin(np.pi * x_array / (2.0 * half_width))
        * np.cos(np.pi * y_array / (2.0 * half_height))
    )
    return angle_to_vector(np.deg2rad(angle_deg))


def synthetic_structural_order_field(
    x: ArrayLike,
    y: ArrayLike,
    *,
    background: float,
    defect_depth: float,
    defect_center_x: float,
    defect_center_y: float,
    defect_width_x: float,
    defect_width_y: float,
) -> FloatArray:
    """Create a bounded beta field with a smooth low-order Gaussian defect."""
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    if not 0.0 <= background <= 1.0:
        raise ValueError("background must lie in [0, 1].")
    if defect_depth < 0.0 or background - defect_depth < 0.0:
        raise ValueError("The defect must keep beta non-negative.")
    if defect_width_x <= 0.0 or defect_width_y <= 0.0:
        raise ValueError("Defect widths must be strictly positive.")

    squared_distance = (
        ((x_array - defect_center_x) / defect_width_x) ** 2
        + ((y_array - defect_center_y) / defect_width_y) ** 2
    )
    beta = background - defect_depth * np.exp(-0.5 * squared_distance)
    return np.clip(beta, 0.0, 1.0)


def compatible_shear_extension_deformation(
    y: ArrayLike,
    time: float,
    *,
    half_height: float,
    maximum_stretch: float,
    maximum_shear: float,
    ramp_duration: float,
) -> FloatArray:
    r"""Return a compatible, area-preserving heterogeneous deformation field.

    The underlying mapping is

    x = lambda X - gamma H/pi cos(pi Y/H),
    y = Y/lambda,

    hence

    F = [[lambda, gamma sin(pi Y/H)], [0, 1/lambda]], det(F) = 1.
    """
    y_array = np.asarray(y, dtype=float)
    if half_height <= 0.0:
        raise ValueError("half_height must be strictly positive.")
    if maximum_stretch < 1.0 or not np.isfinite(maximum_stretch):
        raise ValueError("maximum_stretch must be finite and at least one.")
    if not np.isfinite(maximum_shear):
        raise ValueError("maximum_shear must be finite.")

    load_fraction = smooth_ramp(time, ramp_duration)
    stretch = 1.0 + (maximum_stretch - 1.0) * load_fraction
    shear = maximum_shear * load_fraction

    deformation = np.zeros(y_array.shape + (2, 2), dtype=float)
    deformation[..., 0, 0] = stretch
    deformation[..., 0, 1] = shear * np.sin(np.pi * y_array / half_height)
    deformation[..., 1, 1] = 1.0 / stretch
    return deformation


def nematic_field_statistics(fiber_direction: ArrayLike) -> tuple[float, float]:
    """Return mean nematic angle in radians and orientation coherence in [0, 1]."""
    fiber = normalize_vectors(fiber_direction)
    angle = vector_to_angle(fiber)
    mean_cosine = float(np.mean(np.cos(2.0 * angle)))
    mean_sine = float(np.mean(np.sin(2.0 * angle)))
    mean_angle = float(np.mod(0.5 * np.arctan2(mean_sine, mean_cosine), np.pi))
    coherence = float(np.hypot(mean_cosine, mean_sine))
    return mean_angle, coherence


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


def run_spatial_remodeling(
    config: SpatialSimulationConfig = SpatialSimulationConfig(),
    material: MaterialParameters = MaterialParameters(),
    remodeling: RemodelingParameters = RemodelingParameters(),
    *,
    snapshot_times: ArrayLike | None = None,
) -> SpatialSimulationResult:
    """Run uncoupled local remodeling under a compatible prescribed deformation field.

    Every grid point uses the same constitutive equations as the homogeneous
    model, but local kinematics and internal variables vary in space. Mechanical
    equilibrium and spatial regularization are intentionally outside this
    version's scope.
    """
    x, y = rectangular_grid(config)
    fiber = synthetic_fiber_field(
        x,
        y,
        half_width=config.half_width,
        half_height=config.half_height,
        mean_angle_deg=config.mean_fiber_angle_deg,
        angle_amplitude_deg=config.angle_amplitude_deg,
    )
    beta = synthetic_structural_order_field(
        x,
        y,
        background=config.beta_background,
        defect_depth=config.beta_defect_depth,
        defect_center_x=config.defect_center_x,
        defect_center_y=config.defect_center_y,
        defect_width_x=config.defect_width_x,
        defect_width_y=config.defect_width_y,
    )

    number_of_steps = int(np.floor(config.total_time / config.dt)) + 1
    time = np.arange(number_of_steps, dtype=float) * config.dt
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
    beta_snapshots: list[FloatArray] = []
    beta_equilibrium_snapshots: list[FloatArray] = []
    stimulus_snapshots: list[FloatArray] = []
    energy_snapshots: list[FloatArray] = []
    stress_snapshots: list[FloatArray] = []

    for step, current_time in enumerate(time):
        deformation = compatible_shear_extension_deformation(
            y,
            current_time,
            half_height=config.half_height,
            maximum_stretch=config.maximum_stretch,
            maximum_shear=config.maximum_shear,
            ramp_duration=config.ramp_duration,
        )
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

        nematic_angle, nematic_coherence = nematic_field_statistics(fiber)
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
            deformation_snapshots.append(deformation.copy())
            fiber_snapshots.append(fiber.copy())
            beta_snapshots.append(beta.copy())
            beta_equilibrium_snapshots.append(beta_equilibrium.copy())
            stimulus_snapshots.append(stimulus.copy())
            energy_snapshots.append(energy.copy())
            stress_snapshots.append(stress.copy())

        if step + 1 < number_of_steps:
            activation = hill_activation(
                stimulus,
                half_saturation=remodeling.half_saturation,
                hill_exponent=remodeling.hill_exponent,
            )
            fiber = update_fiber_orientation(
                fiber,
                target,
                rate=remodeling.orientation_rate * activation,
                dt=config.dt,
            )
            beta = update_structural_order(
                beta,
                beta_equilibrium,
                rate=remodeling.order_rate,
                dt=config.dt,
            )

    return SpatialSimulationResult(
        x=x,
        y=y,
        time=time,
        snapshot_time=time[selected_indices],
        deformation_gradient=np.stack(deformation_snapshots),
        fiber_direction=np.stack(fiber_snapshots),
        fiber_angle_deg=np.rad2deg(vector_to_angle(np.stack(fiber_snapshots))),
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
