"""Q4 finite-element equilibrium for multiple discrete fiber families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .architecture import (
    MultiFiberMaterialParameters,
    multifiber_cauchy_stress,
    multifiber_first_piola_stress,
    multifiber_strain_energy_density,
)
from .finite_element import (
    FiniteElementConfig,
    StructuredQuadMesh,
    _displacement_boundary_conditions,
    _reference_quadrature,
)
from .orientation import normalize_vectors

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MultiFiberFiniteElementResult:
    """Equilibrium solution for element-wise multiple fiber families."""

    mesh: StructuredQuadMesh
    displacement: FloatArray
    deformed_nodes: FloatArray
    element_fiber_direction: FloatArray
    element_structural_order: FloatArray
    element_deformation_gradient: FloatArray
    element_jacobian: FloatArray
    element_strain_energy: FloatArray
    element_cauchy_stress: FloatArray
    nodal_internal_force: FloatArray
    free_dof_residual_norm: float
    left_reaction: FloatArray
    right_reaction: FloatArray
    load_factor: FloatArray
    iterations: IntArray
    converged: BoolArray


def _validate_element_multifiber_structure(
    mesh: StructuredQuadMesh,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    material: MultiFiberMaterialParameters,
) -> tuple[FloatArray, FloatArray]:
    fiber = np.asarray(fiber_direction, dtype=float)
    beta = np.asarray(structural_order, dtype=float)
    expected_fiber_shape = (
        mesh.number_of_elements,
        material.number_of_families,
        2,
    )
    expected_beta_shape = (
        mesh.number_of_elements,
        material.number_of_families,
    )
    if fiber.shape != expected_fiber_shape:
        raise ValueError(f"fiber_direction must have shape {expected_fiber_shape}.")
    if beta.shape != expected_beta_shape:
        raise ValueError(f"structural_order must have shape {expected_beta_shape}.")
    if not np.all(np.isfinite(fiber)) or not np.all(np.isfinite(beta)):
        raise ValueError("Element structural fields must be finite.")
    if np.any((beta < 0.0) | (beta > 1.0)):
        raise ValueError("Element structural order must lie in [0, 1].")
    return normalize_vectors(fiber), beta


def assemble_multifiber_internal_energy_and_force(
    mesh: StructuredQuadMesh,
    displacement: ArrayLike,
    element_fiber_direction: ArrayLike,
    element_structural_order: ArrayLike,
    material: MultiFiberMaterialParameters,
    *,
    minimum_jacobian: float = 0.0,
) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Assemble energy, internal force, and element-averaged mechanical fields."""
    displacement_array = np.asarray(displacement, dtype=float)
    if displacement_array.shape != (mesh.number_of_nodes, 2):
        raise ValueError("displacement must have shape (number_of_nodes, 2).")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError("displacement must be finite.")
    fiber, beta = _validate_element_multifiber_structure(
        mesh,
        element_fiber_direction,
        element_structural_order,
        material,
    )
    gradients, measures = _reference_quadrature(mesh)

    force = np.zeros_like(displacement_array)
    total_energy = 0.0
    average_f = np.empty((mesh.number_of_elements, 2, 2), dtype=float)
    average_j = np.empty(mesh.number_of_elements, dtype=float)
    average_energy = np.empty(mesh.number_of_elements, dtype=float)
    average_stress = np.empty((mesh.number_of_elements, 2, 2), dtype=float)
    identity = np.eye(2)

    for element_index, connectivity in enumerate(mesh.elements):
        element_displacement = displacement_array[connectivity]
        element_measure = float(np.sum(measures[element_index]))
        f_accumulator = np.zeros((2, 2), dtype=float)
        j_accumulator = 0.0
        energy_accumulator = 0.0
        stress_accumulator = np.zeros((2, 2), dtype=float)

        for point_index in range(4):
            gradient = gradients[element_index, point_index]
            measure = float(measures[element_index, point_index])
            deformation = identity + element_displacement.T @ gradient
            jacobian = float(np.linalg.det(deformation))
            if jacobian <= minimum_jacobian:
                raise ValueError("Current element Jacobian fell below minimum_jacobian.")

            density = float(
                multifiber_strain_energy_density(
                    deformation,
                    fiber[element_index],
                    beta[element_index],
                    material,
                )
            )
            piola = multifiber_first_piola_stress(
                deformation,
                fiber[element_index],
                beta[element_index],
                material,
            )
            stress = multifiber_cauchy_stress(
                deformation,
                fiber[element_index],
                beta[element_index],
                material,
            )
            element_force = gradient @ piola.T * measure
            np.add.at(force, connectivity, element_force)
            total_energy += density * measure

            f_accumulator += deformation * measure
            j_accumulator += jacobian * measure
            energy_accumulator += density * measure
            stress_accumulator += stress * measure

        average_f[element_index] = f_accumulator / element_measure
        average_j[element_index] = j_accumulator / element_measure
        average_energy[element_index] = energy_accumulator / element_measure
        average_stress[element_index] = stress_accumulator / element_measure

    return total_energy, force, average_f, average_j, average_energy, average_stress


def solve_multifiber_displacement_controlled_equilibrium(
    mesh: StructuredQuadMesh,
    element_fiber_direction: ArrayLike,
    element_structural_order: ArrayLike,
    material: MultiFiberMaterialParameters,
    config: FiniteElementConfig = FiniteElementConfig(),
    *,
    initial_displacement: ArrayLike | None = None,
) -> MultiFiberFiniteElementResult:
    """Solve finite-strain equilibrium for multiple element fiber families."""
    fiber, beta = _validate_element_multifiber_structure(
        mesh,
        element_fiber_direction,
        element_structural_order,
        material,
    )
    number_of_dofs = 2 * mesh.number_of_nodes
    if initial_displacement is None:
        displacement_vector = np.zeros(number_of_dofs, dtype=float)
    else:
        initial = np.asarray(initial_displacement, dtype=float)
        if initial.shape != (mesh.number_of_nodes, 2):
            raise ValueError(
                "initial_displacement must have shape (number_of_nodes, 2)."
            )
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_displacement must be finite.")
        displacement_vector = initial.ravel().copy()

    load_factors = np.linspace(0.0, 1.0, config.load_steps + 1)[1:]
    iterations = np.zeros(config.load_steps, dtype=np.int64)
    converged = np.zeros(config.load_steps, dtype=bool)
    final_free_dofs: IntArray | None = None
    left_nodes: IntArray | None = None
    right_nodes: IntArray | None = None
    previous_load_factor = 0.0

    for step_index, load_factor in enumerate(load_factors):
        if previous_load_factor > 0.0:
            displacement_vector *= float(load_factor / previous_load_factor)
        extension = config.axial_extension * mesh.width * float(load_factor)
        fixed_dofs, fixed_values, left_nodes, right_nodes, _ = (
            _displacement_boundary_conditions(mesh, extension)
        )
        all_dofs = np.arange(number_of_dofs, dtype=np.int64)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)
        displacement_vector[fixed_dofs] = fixed_values
        initial_free = displacement_vector[free_dofs].copy()

        def objective(free_values: FloatArray) -> tuple[float, FloatArray]:
            trial = displacement_vector.copy()
            trial[fixed_dofs] = fixed_values
            trial[free_dofs] = free_values
            try:
                energy, force, *_ = assemble_multifiber_internal_energy_and_force(
                    mesh,
                    trial.reshape((-1, 2)),
                    fiber,
                    beta,
                    material,
                    minimum_jacobian=config.minimum_jacobian,
                )
            except ValueError:
                return 1.0e30, np.zeros_like(free_values)
            return energy, force.ravel()[free_dofs]

        optimization = minimize(
            objective,
            initial_free,
            method="BFGS",
            jac=True,
            options={
                "gtol": config.gradient_tolerance,
                "maxiter": config.maximum_iterations,
            },
        )
        displacement_vector[fixed_dofs] = fixed_values
        displacement_vector[free_dofs] = optimization.x
        iterations[step_index] = int(optimization.nit)

        _, force, _, jacobian, _, _ = (
            assemble_multifiber_internal_energy_and_force(
                mesh,
                displacement_vector.reshape((-1, 2)),
                fiber,
                beta,
                material,
                minimum_jacobian=config.minimum_jacobian,
            )
        )
        residual_norm = float(np.linalg.norm(force.ravel()[free_dofs], ord=np.inf))
        converged[step_index] = bool(
            optimization.success or residual_norm <= 10.0 * config.gradient_tolerance
        )
        if not converged[step_index]:
            raise RuntimeError(
                f"Multi-fiber FE solve failed at load step {step_index + 1}: "
                f"{optimization.message}; residual={residual_norm:.3e}; "
                f"min(J)={np.min(jacobian):.6f}."
            )
        final_free_dofs = free_dofs
        previous_load_factor = float(load_factor)

    assert final_free_dofs is not None
    assert left_nodes is not None and right_nodes is not None
    displacement = displacement_vector.reshape((-1, 2))
    _, internal_force, deformation, jacobian, density, stress = (
        assemble_multifiber_internal_energy_and_force(
            mesh,
            displacement,
            fiber,
            beta,
            material,
            minimum_jacobian=config.minimum_jacobian,
        )
    )
    residual_norm = float(
        np.linalg.norm(internal_force.ravel()[final_free_dofs], ord=np.inf)
    )

    return MultiFiberFiniteElementResult(
        mesh=mesh,
        displacement=displacement,
        deformed_nodes=mesh.nodes + displacement,
        element_fiber_direction=fiber,
        element_structural_order=beta,
        element_deformation_gradient=deformation,
        element_jacobian=jacobian,
        element_strain_energy=density,
        element_cauchy_stress=stress,
        nodal_internal_force=internal_force,
        free_dof_residual_norm=residual_norm,
        left_reaction=np.sum(internal_force[left_nodes], axis=0),
        right_reaction=np.sum(internal_force[right_nodes], axis=0),
        load_factor=load_factors,
        iterations=iterations,
        converged=converged,
    )
