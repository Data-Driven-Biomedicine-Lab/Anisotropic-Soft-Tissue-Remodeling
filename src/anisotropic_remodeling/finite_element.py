"""Minimal total-Lagrangian Q4 finite-element equilibrium solver.

The module solves two-dimensional, displacement-controlled hyperelastic
boundary-value problems with spatially varying fiber direction and structural
order. It is intentionally small and transparent: four-node bilinear
quadrilaterals, 2x2 Gauss integration, analytical internal forces, and
incremental energy minimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .material import MaterialParameters, cauchy_stress, first_piola_stress, strain_energy_density
from .orientation import normalize_vectors, vector_to_angle

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class StructuredQuadMesh:
    """A rectangular mesh of counter-clockwise four-node quadrilaterals."""

    nodes: FloatArray
    elements: IntArray
    number_of_elements_x: int
    number_of_elements_y: int
    width: float
    height: float

    @property
    def number_of_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def number_of_elements(self) -> int:
        return int(self.elements.shape[0])


@dataclass(frozen=True, slots=True)
class FiniteElementConfig:
    """Displacement-control and nonlinear-solver settings."""

    axial_extension: float = 0.12
    load_steps: int = 6
    gradient_tolerance: float = 1.0e-7
    maximum_iterations: int = 500
    minimum_jacobian: float = 0.20

    def __post_init__(self) -> None:
        if not np.isfinite(self.axial_extension) or self.axial_extension <= 0.0:
            raise ValueError("axial_extension must be finite and strictly positive.")
        if self.load_steps < 1:
            raise ValueError("load_steps must be at least one.")
        if not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0.0:
            raise ValueError("gradient_tolerance must be finite and strictly positive.")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be at least one.")
        if not np.isfinite(self.minimum_jacobian) or not 0.0 < self.minimum_jacobian < 1.0:
            raise ValueError("minimum_jacobian must lie in (0, 1).")


@dataclass(frozen=True, slots=True)
class FiniteElementResult:
    """Equilibrium solution and element-level mechanical fields."""

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


_GAUSS_COORDINATE = 1.0 / np.sqrt(3.0)
_GAUSS_POINTS = np.array(
    [
        [-_GAUSS_COORDINATE, -_GAUSS_COORDINATE],
        [_GAUSS_COORDINATE, -_GAUSS_COORDINATE],
        [_GAUSS_COORDINATE, _GAUSS_COORDINATE],
        [-_GAUSS_COORDINATE, _GAUSS_COORDINATE],
    ],
    dtype=float,
)


def rectangular_quad_mesh(
    number_of_elements_x: int,
    number_of_elements_y: int,
    *,
    width: float = 2.0,
    height: float = 1.0,
) -> StructuredQuadMesh:
    """Create a structured rectangular Q4 mesh."""
    if number_of_elements_x < 1 or number_of_elements_y < 1:
        raise ValueError("The mesh must contain at least one element in each direction.")
    if not np.isfinite(width) or width <= 0.0 or not np.isfinite(height) or height <= 0.0:
        raise ValueError("width and height must be finite and strictly positive.")

    x = np.linspace(0.0, width, number_of_elements_x + 1)
    y = np.linspace(0.0, height, number_of_elements_y + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))

    elements: list[list[int]] = []
    row = number_of_elements_x + 1
    for j in range(number_of_elements_y):
        for i in range(number_of_elements_x):
            lower_left = j * row + i
            elements.append(
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + row + 1,
                    lower_left + row,
                ]
            )

    return StructuredQuadMesh(
        nodes=nodes,
        elements=np.asarray(elements, dtype=np.int64),
        number_of_elements_x=number_of_elements_x,
        number_of_elements_y=number_of_elements_y,
        width=float(width),
        height=float(height),
    )


def element_centroids(mesh: StructuredQuadMesh) -> FloatArray:
    """Return reference centroids of all elements."""
    return np.mean(mesh.nodes[mesh.elements], axis=1)


def _natural_shape_gradients(xi: float, eta: float) -> FloatArray:
    return 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=float,
    )


def _reference_quadrature(mesh: StructuredQuadMesh) -> tuple[FloatArray, FloatArray]:
    number_of_elements = mesh.number_of_elements
    gradients = np.empty((number_of_elements, 4, 4, 2), dtype=float)
    measures = np.empty((number_of_elements, 4), dtype=float)

    for element_index, connectivity in enumerate(mesh.elements):
        coordinates = mesh.nodes[connectivity]
        for point_index, (xi, eta) in enumerate(_GAUSS_POINTS):
            natural_gradient = _natural_shape_gradients(float(xi), float(eta))
            reference_jacobian = coordinates.T @ natural_gradient
            determinant = float(np.linalg.det(reference_jacobian))
            if determinant <= 0.0:
                raise ValueError("Mesh elements must have positive reference Jacobians.")
            gradients[element_index, point_index] = natural_gradient @ np.linalg.inv(
                reference_jacobian
            )
            measures[element_index, point_index] = determinant
    return gradients, measures


def _validate_element_structure(
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
        raise ValueError("Element structural fields must be finite.")
    if np.any((beta < 0.0) | (beta > 1.0)):
        raise ValueError("Element structural order must lie in [0, 1].")
    return normalize_vectors(fiber), beta


def assemble_internal_energy_and_force(
    mesh: StructuredQuadMesh,
    displacement: ArrayLike,
    element_fiber_direction: ArrayLike,
    element_structural_order: ArrayLike,
    material: MaterialParameters = MaterialParameters(),
    *,
    minimum_jacobian: float = 0.0,
) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Assemble total strain energy and its analytical nodal gradient.

    Returns total energy, nodal internal force, and element-averaged fields
    ``F``, ``J``, strain-energy density, and Cauchy stress.
    """
    displacement_array = np.asarray(displacement, dtype=float)
    if displacement_array.shape != (mesh.number_of_nodes, 2):
        raise ValueError("displacement must have shape (number_of_nodes, 2).")
    if not np.all(np.isfinite(displacement_array)):
        raise ValueError("displacement must be finite.")
    fiber, beta = _validate_element_structure(
        mesh, element_fiber_direction, element_structural_order
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
                strain_energy_density(deformation, fiber[element_index], beta[element_index], material)
            )
            piola = first_piola_stress(
                deformation, fiber[element_index], beta[element_index], material
            )
            stress = cauchy_stress(
                deformation, fiber[element_index], beta[element_index], material
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


def _displacement_boundary_conditions(
    mesh: StructuredQuadMesh,
    extension: float,
) -> tuple[IntArray, FloatArray, IntArray, IntArray, int]:
    tolerance = 1.0e-12 * max(mesh.width, mesh.height, 1.0)
    left_nodes = np.flatnonzero(np.isclose(mesh.nodes[:, 0], 0.0, atol=tolerance))
    right_nodes = np.flatnonzero(np.isclose(mesh.nodes[:, 0], mesh.width, atol=tolerance))
    lower_left = int(
        np.argmin(np.linalg.norm(mesh.nodes - np.array([0.0, 0.0]), axis=1))
    )

    fixed_dofs = np.concatenate((2 * left_nodes, 2 * right_nodes, np.array([2 * lower_left + 1])))
    fixed_values = np.concatenate(
        (
            np.zeros(left_nodes.size),
            np.full(right_nodes.size, extension),
            np.array([0.0]),
        )
    )
    unique_dofs, unique_indices = np.unique(fixed_dofs, return_index=True)
    return unique_dofs, fixed_values[unique_indices], left_nodes, right_nodes, lower_left


def solve_displacement_controlled_equilibrium(
    mesh: StructuredQuadMesh,
    element_fiber_direction: ArrayLike,
    element_structural_order: ArrayLike,
    material: MaterialParameters = MaterialParameters(),
    config: FiniteElementConfig = FiniteElementConfig(),
    *,
    initial_displacement: ArrayLike | None = None,
) -> FiniteElementResult:
    """Solve a displacement-controlled finite-strain equilibrium problem.

    The left and right edges receive prescribed horizontal displacements. One
    lower-left vertical degree of freedom removes rigid translation; all other
    vertical degrees of freedom are traction-free. ``initial_displacement`` can
    be supplied as a warm start, which is useful when the material structure is
    updated incrementally during remodeling.
    """
    fiber, beta = _validate_element_structure(
        mesh, element_fiber_direction, element_structural_order
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

    final_fixed_dofs: IntArray | None = None
    final_free_dofs: IntArray | None = None
    left_nodes: IntArray | None = None
    right_nodes: IntArray | None = None

    previous_load_factor = 0.0
    for step_index, load_factor in enumerate(load_factors):
        if previous_load_factor > 0.0:
            displacement_vector *= float(load_factor / previous_load_factor)
        extension = config.axial_extension * mesh.width * float(load_factor)
        fixed_dofs, fixed_values, left_nodes, right_nodes, _ = _displacement_boundary_conditions(
            mesh, extension
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
                energy, force, *_ = assemble_internal_energy_and_force(
                    mesh,
                    trial.reshape((-1, 2)),
                    fiber,
                    beta,
                    material,
                    minimum_jacobian=config.minimum_jacobian,
                )
            except ValueError:
                # A smooth line search starting from the previous load step
                # should rarely enter this branch. The large finite value keeps
                # the optimizer away from inverted configurations.
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

        _, force, _, jacobian, _, _ = assemble_internal_energy_and_force(
            mesh,
            displacement_vector.reshape((-1, 2)),
            fiber,
            beta,
            material,
            minimum_jacobian=config.minimum_jacobian,
        )
        residual_norm = float(np.linalg.norm(force.ravel()[free_dofs], ord=np.inf))
        converged[step_index] = bool(
            optimization.success or residual_norm <= 10.0 * config.gradient_tolerance
        )
        if not converged[step_index]:
            raise RuntimeError(
                f"Finite-element solve failed at load step {step_index + 1}: "
                f"{optimization.message}; residual={residual_norm:.3e}; "
                f"min(J)={np.min(jacobian):.6f}."
            )
        final_fixed_dofs = fixed_dofs
        final_free_dofs = free_dofs
        previous_load_factor = float(load_factor)

    assert final_fixed_dofs is not None and final_free_dofs is not None
    assert left_nodes is not None and right_nodes is not None
    displacement = displacement_vector.reshape((-1, 2))
    energy, internal_force, deformation, jacobian, density, stress = (
        assemble_internal_energy_and_force(
            mesh,
            displacement,
            fiber,
            beta,
            material,
            minimum_jacobian=config.minimum_jacobian,
        )
    )
    del energy, final_fixed_dofs
    residual_norm = float(np.linalg.norm(internal_force.ravel()[final_free_dofs], ord=np.inf))

    return FiniteElementResult(
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


def sample_nematic_image_to_elements(
    points: ArrayLike,
    image_x: ArrayLike,
    image_y: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    valid_mask: ArrayLike,
    *,
    neighbors: int = 8,
    distance_power: float = 2.0,
) -> tuple[FloatArray, FloatArray]:
    """Sample masked image fields at element points using nematic-safe k-NN.

    Fiber directions are averaged in doubled-angle space, preserving the
    equivalence ``a == -a``. Structural order is averaged with the same
    inverse-distance weights.
    """
    query = np.asarray(points, dtype=float)
    x = np.asarray(image_x, dtype=float)
    y = np.asarray(image_y, dtype=float)
    fiber = np.asarray(fiber_direction, dtype=float)
    beta = np.asarray(structural_order, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)

    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("points must have shape (number_of_points, 2).")
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("image_x and image_y must be matching two-dimensional maps.")
    if fiber.shape != x.shape + (2,) or beta.shape != x.shape or mask.shape != x.shape:
        raise ValueError("Image structural fields do not match the coordinate maps.")
    if neighbors < 1:
        raise ValueError("neighbors must be at least one.")
    if distance_power <= 0.0 or not np.isfinite(distance_power):
        raise ValueError("distance_power must be finite and strictly positive.")
    if not np.any(mask):
        raise ValueError("valid_mask must contain at least one valid pixel.")

    coordinates = np.column_stack((x[mask], y[mask]))
    valid_fiber = normalize_vectors(fiber[mask])
    valid_beta = beta[mask]
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(valid_beta)):
        raise ValueError("Valid image data must be finite.")

    tree = cKDTree(coordinates)
    k = min(int(neighbors), coordinates.shape[0])
    distances, indices = tree.query(query, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    epsilon = np.finfo(float).eps
    weights = 1.0 / np.maximum(distances, epsilon) ** distance_power
    exact = distances <= 10.0 * epsilon
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        weights[exact_rows] = exact[exact_rows].astype(float)
    weights /= np.sum(weights, axis=1, keepdims=True)

    angles = vector_to_angle(valid_fiber[indices])
    cosine = np.sum(weights * np.cos(2.0 * angles), axis=1)
    sine = np.sum(weights * np.sin(2.0 * angles), axis=1)
    sampled_angle = 0.5 * np.arctan2(sine, cosine)
    sampled_fiber = np.stack((np.cos(sampled_angle), np.sin(sampled_angle)), axis=-1)
    sampled_beta = np.sum(weights * valid_beta[indices], axis=1)
    return normalize_vectors(sampled_fiber), np.clip(sampled_beta, 0.0, 1.0)
