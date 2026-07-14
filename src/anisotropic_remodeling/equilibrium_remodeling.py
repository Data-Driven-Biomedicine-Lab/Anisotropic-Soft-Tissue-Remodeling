"""Staggered finite-element equilibrium and structural remodeling.

At each remodeling time point, the displacement-controlled finite-element
boundary-value problem is solved for the current material structure. Element
fiber directions and structural order are then updated from the equilibrium
deformation field before mechanical equilibrium is solved again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .finite_element import (
    FiniteElementConfig,
    StructuredQuadMesh,
    solve_displacement_controlled_equilibrium,
)
from .material import MaterialParameters
from .orientation import normalize_vectors, vector_to_angle
from .remodeling import RemodelingParameters, update_fiber_orientation, update_structural_order
from .stimuli import (
    directional_stretch_stimulus,
    equilibrium_structural_order,
    hill_activation,
    principal_stretch_direction,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EquilibriumRemodelingConfig:
    """Time integration and nonlinear equilibrium settings."""

    total_time: float = 6.0
    dt: float = 0.5
    axial_extension: float = 0.06
    initial_load_steps: int = 3
    subsequent_load_steps: int = 1
    gradient_tolerance: float = 2.0e-7
    maximum_iterations: int = 500
    minimum_jacobian: float = 0.20

    def __post_init__(self) -> None:
        if not np.isfinite(self.total_time) or self.total_time <= 0.0:
            raise ValueError("total_time must be finite and strictly positive.")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive.")
        if not np.isfinite(self.axial_extension) or self.axial_extension <= 0.0:
            raise ValueError("axial_extension must be finite and strictly positive.")
        if self.initial_load_steps < 1 or self.subsequent_load_steps < 1:
            raise ValueError("Load-step counts must be at least one.")
        if not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0.0:
            raise ValueError("gradient_tolerance must be finite and strictly positive.")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be at least one.")
        if not np.isfinite(self.minimum_jacobian) or not 0.0 < self.minimum_jacobian < 1.0:
            raise ValueError("minimum_jacobian must lie in (0, 1).")


@dataclass(frozen=True, slots=True)
class EquilibriumRemodelingResult:
    """Complete history of the staggered equilibrium-remodeling calculation."""

    mesh: StructuredQuadMesh
    time: FloatArray
    displacement: FloatArray
    fiber_direction: FloatArray
    fiber_angle_deg: FloatArray
    structural_order: FloatArray
    equilibrium_order: FloatArray
    stimulus: FloatArray
    target_direction: FloatArray
    element_deformation_gradient: FloatArray
    element_jacobian: FloatArray
    element_strain_energy: FloatArray
    element_cauchy_stress: FloatArray
    left_reaction: FloatArray
    right_reaction: FloatArray
    free_dof_residual_norm: FloatArray
    iterations: IntArray
    converged: BoolArray
    mean_structural_order: FloatArray
    mean_equilibrium_order: FloatArray
    mean_stimulus: FloatArray
    mean_strain_energy: FloatArray
    mean_cauchy_stress_xx: FloatArray
    mean_target_alignment: FloatArray
    orientation_coherence: FloatArray


def _time_grid(total_time: float, dt: float) -> FloatArray:
    number_of_intervals = int(np.ceil(total_time / dt))
    return np.linspace(0.0, total_time, number_of_intervals + 1, dtype=float)


def _validate_initial_structure(
    mesh: StructuredQuadMesh,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    fiber = np.asarray(fiber_direction, dtype=float)
    beta = np.asarray(structural_order, dtype=float)
    if fiber.shape != (mesh.number_of_elements, 2):
        raise ValueError("fiber_direction must have shape (number_of_elements, 2).")
    if beta.shape != (mesh.number_of_elements,):
        raise ValueError("structural_order must have shape (number_of_elements,).")
    if not np.all(np.isfinite(fiber)) or not np.all(np.isfinite(beta)):
        raise ValueError("Initial structural fields must be finite.")
    if np.any((beta < 0.0) | (beta > 1.0)):
        raise ValueError("Initial structural order must lie in [0, 1].")
    return normalize_vectors(fiber), beta.copy()


def _nematic_coherence(fiber_direction: FloatArray) -> float:
    angle = vector_to_angle(fiber_direction)
    cosine = float(np.mean(np.cos(2.0 * angle)))
    sine = float(np.mean(np.sin(2.0 * angle)))
    return float(np.hypot(cosine, sine))


def run_equilibrium_remodeling(
    mesh: StructuredQuadMesh,
    initial_fiber_direction: ArrayLike,
    initial_structural_order: ArrayLike,
    material: MaterialParameters = MaterialParameters(),
    remodeling: RemodelingParameters = RemodelingParameters(),
    config: EquilibriumRemodelingConfig = EquilibriumRemodelingConfig(),
) -> EquilibriumRemodelingResult:
    """Run a staggered finite-element equilibrium-remodeling simulation.

    The applied displacement is held fixed throughout remodeling. At each time
    point the mechanical equilibrium problem is solved for the current element
    structure. The resulting element deformation gradients define the local
    target direction and directional-stretch stimulus used in the exact kinetic
    updates over the following time interval.
    """
    fiber, beta = _validate_initial_structure(
        mesh,
        initial_fiber_direction,
        initial_structural_order,
    )
    time = _time_grid(config.total_time, config.dt)
    number_of_times = time.size
    number_of_elements = mesh.number_of_elements
    number_of_nodes = mesh.number_of_nodes

    displacement_history = np.empty((number_of_times, number_of_nodes, 2), dtype=float)
    fiber_history = np.empty((number_of_times, number_of_elements, 2), dtype=float)
    angle_history = np.empty((number_of_times, number_of_elements), dtype=float)
    beta_history = np.empty((number_of_times, number_of_elements), dtype=float)
    beta_equilibrium_history = np.empty_like(beta_history)
    stimulus_history = np.empty_like(beta_history)
    target_history = np.empty_like(fiber_history)
    deformation_history = np.empty((number_of_times, number_of_elements, 2, 2), dtype=float)
    jacobian_history = np.empty((number_of_times, number_of_elements), dtype=float)
    energy_history = np.empty_like(beta_history)
    stress_history = np.empty((number_of_times, number_of_elements, 2, 2), dtype=float)
    left_reaction = np.empty((number_of_times, 2), dtype=float)
    right_reaction = np.empty((number_of_times, 2), dtype=float)
    residual = np.empty(number_of_times, dtype=float)
    iterations = np.empty(number_of_times, dtype=np.int64)
    converged = np.empty(number_of_times, dtype=bool)

    mean_beta = np.empty(number_of_times, dtype=float)
    mean_beta_equilibrium = np.empty(number_of_times, dtype=float)
    mean_stimulus = np.empty(number_of_times, dtype=float)
    mean_energy = np.empty(number_of_times, dtype=float)
    mean_sigma_xx = np.empty(number_of_times, dtype=float)
    mean_alignment = np.empty(number_of_times, dtype=float)
    coherence = np.empty(number_of_times, dtype=float)

    warm_start: FloatArray | None = None
    for time_index, current_time in enumerate(time):
        del current_time
        load_steps = (
            config.initial_load_steps if time_index == 0 else config.subsequent_load_steps
        )
        finite_element_config = FiniteElementConfig(
            axial_extension=config.axial_extension,
            load_steps=load_steps,
            gradient_tolerance=config.gradient_tolerance,
            maximum_iterations=config.maximum_iterations,
            minimum_jacobian=config.minimum_jacobian,
        )
        equilibrium = solve_displacement_controlled_equilibrium(
            mesh,
            fiber,
            beta,
            material,
            finite_element_config,
            initial_displacement=warm_start,
        )
        warm_start = equilibrium.displacement.copy()

        deformation = equilibrium.element_deformation_gradient
        stimulus = directional_stretch_stimulus(deformation)
        target = principal_stretch_direction(deformation)
        beta_equilibrium = equilibrium_structural_order(
            stimulus,
            beta_min=remodeling.beta_min,
            beta_max=remodeling.beta_max,
            half_saturation=remodeling.half_saturation,
            hill_exponent=remodeling.hill_exponent,
        )
        alignment = np.einsum("ei,ei->e", fiber, target) ** 2

        displacement_history[time_index] = equilibrium.displacement
        fiber_history[time_index] = fiber
        angle_history[time_index] = np.rad2deg(vector_to_angle(fiber))
        beta_history[time_index] = beta
        beta_equilibrium_history[time_index] = beta_equilibrium
        stimulus_history[time_index] = stimulus
        target_history[time_index] = target
        deformation_history[time_index] = deformation
        jacobian_history[time_index] = equilibrium.element_jacobian
        energy_history[time_index] = equilibrium.element_strain_energy
        stress_history[time_index] = equilibrium.element_cauchy_stress
        left_reaction[time_index] = equilibrium.left_reaction
        right_reaction[time_index] = equilibrium.right_reaction
        residual[time_index] = equilibrium.free_dof_residual_norm
        iterations[time_index] = int(np.sum(equilibrium.iterations))
        converged[time_index] = bool(np.all(equilibrium.converged))
        mean_beta[time_index] = float(np.mean(beta))
        mean_beta_equilibrium[time_index] = float(np.mean(beta_equilibrium))
        mean_stimulus[time_index] = float(np.mean(stimulus))
        mean_energy[time_index] = float(np.mean(equilibrium.element_strain_energy))
        mean_sigma_xx[time_index] = float(
            np.mean(equilibrium.element_cauchy_stress[:, 0, 0])
        )
        mean_alignment[time_index] = float(np.mean(alignment))
        coherence[time_index] = _nematic_coherence(fiber)

        if time_index == number_of_times - 1:
            continue
        step_size = float(time[time_index + 1] - time[time_index])
        orientation_activation = hill_activation(
            stimulus,
            half_saturation=remodeling.half_saturation,
            hill_exponent=remodeling.hill_exponent,
        )
        fiber = update_fiber_orientation(
            fiber,
            target,
            rate=remodeling.orientation_rate * orientation_activation,
            dt=step_size,
        )
        beta = update_structural_order(
            beta,
            beta_equilibrium,
            rate=remodeling.order_rate,
            dt=step_size,
        )

    return EquilibriumRemodelingResult(
        mesh=mesh,
        time=time,
        displacement=displacement_history,
        fiber_direction=fiber_history,
        fiber_angle_deg=angle_history,
        structural_order=beta_history,
        equilibrium_order=beta_equilibrium_history,
        stimulus=stimulus_history,
        target_direction=target_history,
        element_deformation_gradient=deformation_history,
        element_jacobian=jacobian_history,
        element_strain_energy=energy_history,
        element_cauchy_stress=stress_history,
        left_reaction=left_reaction,
        right_reaction=right_reaction,
        free_dof_residual_norm=residual,
        iterations=iterations,
        converged=converged,
        mean_structural_order=mean_beta,
        mean_equilibrium_order=mean_beta_equilibrium,
        mean_stimulus=mean_stimulus,
        mean_strain_energy=mean_energy,
        mean_cauchy_stress_xx=mean_sigma_xx,
        mean_target_alignment=mean_alignment,
        orientation_coherence=coherence,
    )
