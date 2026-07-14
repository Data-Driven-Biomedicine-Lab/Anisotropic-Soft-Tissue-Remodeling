"""Multiple fiber families, orientation distributions, and graph regularization.

This module extends the minimal single-family formulation without changing its
public behavior. A tissue point may contain several undirected fiber families,
each with its own direction, structural order, stiffness, and mixture weight.

Spatial regularization is defined on the element-adjacency graph. Scalar fields
are smoothed by an implicit graph-Laplacian step. Nematic directions are
regularized in doubled-angle space, preserving the equivalence ``a == -a``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.special import i0, i1

from .finite_element import StructuredQuadMesh
from .orientation import angle_to_vector, normalize_vectors, vector_to_angle

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class MultiFiberMaterialParameters:
    """Material parameters for a matrix reinforced by discrete fiber families.

    Parameters
    ----------
    mu, kappa:
        Compressible neo-Hookean matrix parameters.
    k1, k2:
        Per-family exponential-fiber parameters.
    family_weights:
        Non-negative mixture weights. They are normalized internally to sum to
        one, so the total fiber scale remains interpretable when the number of
        families changes.
    """

    mu: float = 10.0
    kappa: float = 1000.0
    k1: tuple[float, ...] = (2.0, 2.0)
    k2: tuple[float, ...] = (5.0, 5.0)
    family_weights: tuple[float, ...] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError("mu must be finite and strictly positive.")
        if not np.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError("kappa must be finite and strictly positive.")
        if len(self.k1) == 0:
            raise ValueError("At least one fiber family is required.")
        if not (len(self.k1) == len(self.k2) == len(self.family_weights)):
            raise ValueError("k1, k2, and family_weights must have equal lengths.")
        for name, values in (
            ("k1", self.k1),
            ("k2", self.k2),
        ):
            array = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"{name} values must be finite and strictly positive.")
        weights = np.asarray(self.family_weights, dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("family_weights must be finite and non-negative.")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("At least one family weight must be strictly positive.")

    @property
    def number_of_families(self) -> int:
        return len(self.k1)

    @property
    def normalized_weights(self) -> FloatArray:
        weights = np.asarray(self.family_weights, dtype=float)
        return weights / np.sum(weights)

    @property
    def k1_array(self) -> FloatArray:
        return np.asarray(self.k1, dtype=float)

    @property
    def k2_array(self) -> FloatArray:
        return np.asarray(self.k2, dtype=float)


def _validate_multifiber_inputs(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MultiFiberMaterialParameters,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("deformation_gradient must have shape (..., 2, 2).")
    if not np.all(np.isfinite(deformation)):
        raise ValueError("deformation_gradient must be finite.")
    determinant = np.linalg.det(deformation)
    if np.any(determinant <= 0.0):
        raise ValueError("deformation_gradient must satisfy det(F) > 0.")

    fiber = np.asarray(fiber_direction, dtype=float)
    if fiber.shape[-1] != 2 or fiber.ndim < 2:
        raise ValueError("fiber_direction must have shape (..., families, 2).")
    if fiber.shape[-2] != parameters.number_of_families:
        raise ValueError("The fiber-family axis does not match the material parameters.")
    fiber = normalize_vectors(fiber)

    order = np.asarray(structural_order, dtype=float)
    if order.shape[-1:] != (parameters.number_of_families,):
        raise ValueError("structural_order must have shape (..., families).")
    if not np.all(np.isfinite(order)):
        raise ValueError("structural_order must be finite.")
    if np.any((order < 0.0) | (order > 1.0)):
        raise ValueError("structural_order values must lie in [0, 1].")

    batch_shape = np.broadcast_shapes(
        deformation.shape[:-2],
        fiber.shape[:-2],
        order.shape[:-1],
    )
    deformation = np.broadcast_to(deformation, batch_shape + (2, 2))
    fiber = np.broadcast_to(
        fiber,
        batch_shape + (parameters.number_of_families, 2),
    )
    order = np.broadcast_to(
        order,
        batch_shape + (parameters.number_of_families,),
    )
    determinant = np.broadcast_to(determinant, batch_shape)
    return deformation, fiber, order, determinant


def multifiber_invariants(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``J``, ``I1``, and one ``I4`` value per fiber family."""
    deformation = np.asarray(deformation_gradient, dtype=float)
    fiber = normalize_vectors(fiber_direction)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("deformation_gradient must have shape (..., 2, 2).")
    if fiber.shape[-1] != 2 or fiber.ndim < 2:
        raise ValueError("fiber_direction must have shape (..., families, 2).")
    determinant = np.linalg.det(deformation)
    if np.any(determinant <= 0.0):
        raise ValueError("deformation_gradient must satisfy det(F) > 0.")

    batch_shape = np.broadcast_shapes(
        deformation.shape[:-2],
        fiber.shape[:-2],
    )
    deformation = np.broadcast_to(deformation, batch_shape + (2, 2))
    fiber = np.broadcast_to(fiber, batch_shape + fiber.shape[-2:])
    determinant = np.broadcast_to(determinant, batch_shape)

    right_cauchy_green = np.einsum("...ki,...kj->...ij", deformation, deformation)
    first_invariant = np.trace(right_cauchy_green, axis1=-2, axis2=-1)
    fourth_invariant = np.einsum(
        "...mi,...ij,...mj->...m",
        fiber,
        right_cauchy_green,
        fiber,
    )
    return determinant, first_invariant, fourth_invariant


def multifiber_strain_energy_density(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MultiFiberMaterialParameters = MultiFiberMaterialParameters(),
) -> FloatArray:
    """Return matrix plus weighted tension-only fiber energy."""
    deformation, fiber, order, determinant = _validate_multifiber_inputs(
        deformation_gradient,
        fiber_direction,
        structural_order,
        parameters,
    )
    _, first_invariant, fourth_invariant = multifiber_invariants(
        deformation,
        fiber,
    )
    logarithmic_jacobian = np.log(determinant)
    matrix_energy = (
        0.5 * parameters.mu * (first_invariant - 2.0 - 2.0 * logarithmic_jacobian)
        + 0.5 * parameters.kappa * logarithmic_jacobian**2
    )

    extension = np.maximum(fourth_invariant - 1.0, 0.0)
    k1 = parameters.k1_array
    k2 = parameters.k2_array
    weights = parameters.normalized_weights
    family_energy = (
        weights
        * order
        * k1
        / (2.0 * k2)
        * np.expm1(k2 * extension**2)
    )
    return matrix_energy + np.sum(family_energy, axis=-1)


def multifiber_first_piola_stress(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MultiFiberMaterialParameters = MultiFiberMaterialParameters(),
) -> FloatArray:
    """Return the analytical first Piola stress for all fiber families."""
    deformation, fiber, order, determinant = _validate_multifiber_inputs(
        deformation_gradient,
        fiber_direction,
        structural_order,
        parameters,
    )
    inverse_transpose = np.swapaxes(np.linalg.inv(deformation), -1, -2)
    logarithmic_jacobian = np.log(determinant)
    matrix_stress = (
        parameters.mu * (deformation - inverse_transpose)
        + parameters.kappa * logarithmic_jacobian[..., None, None] * inverse_transpose
    )

    right_cauchy_green = np.einsum("...ki,...kj->...ij", deformation, deformation)
    fourth_invariant = np.einsum(
        "...mi,...ij,...mj->...m",
        fiber,
        right_cauchy_green,
        fiber,
    )
    extension = np.maximum(fourth_invariant - 1.0, 0.0)
    multiplier = (
        2.0
        * parameters.normalized_weights
        * order
        * parameters.k1_array
        * extension
        * np.exp(parameters.k2_array * extension**2)
    )
    deformed_fiber = np.einsum("...ij,...mj->...mi", deformation, fiber)
    family_stress = multiplier[..., :, None, None] * np.einsum(
        "...mi,...mj->...mij",
        deformed_fiber,
        fiber,
    )
    return matrix_stress + np.sum(family_stress, axis=-3)


def multifiber_cauchy_stress(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MultiFiberMaterialParameters = MultiFiberMaterialParameters(),
) -> FloatArray:
    """Return ``sigma = J^-1 P F^T`` for the multi-family material."""
    deformation, fiber, order, determinant = _validate_multifiber_inputs(
        deformation_gradient,
        fiber_direction,
        structural_order,
        parameters,
    )
    piola = multifiber_first_piola_stress(
        deformation,
        fiber,
        order,
        parameters,
    )
    return np.einsum("...ik,...jk->...ij", piola, deformation) / determinant[..., None, None]


def discrete_nematic_distribution(
    mean_angle: float,
    concentration: float,
    *,
    number_of_directions: int = 72,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Discretize a pi-periodic von Mises orientation distribution.

    The density is proportional to ``exp(kappa*cos(2*(theta-mean)))`` on
    ``theta in [0, pi)``. Returned weights sum to one.
    """
    if not np.isfinite(mean_angle):
        raise ValueError("mean_angle must be finite.")
    if not np.isfinite(concentration) or concentration < 0.0:
        raise ValueError("concentration must be finite and non-negative.")
    if number_of_directions < 4:
        raise ValueError("number_of_directions must be at least four.")

    angles = (np.arange(number_of_directions, dtype=float) + 0.5) * (
        np.pi / number_of_directions
    )
    log_weights = concentration * np.cos(2.0 * (angles - mean_angle))
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return angles, angle_to_vector(angles), weights


def theoretical_nematic_coherence(concentration: float) -> float:
    """Return ``I1(kappa)/I0(kappa)`` for the nematic von Mises density."""
    if not np.isfinite(concentration) or concentration < 0.0:
        raise ValueError("concentration must be finite and non-negative.")
    return float(i1(concentration) / i0(concentration))


def discrete_nematic_coherence(
    fiber_direction: ArrayLike,
    weights: ArrayLike,
) -> float:
    """Return the weighted magnitude of the doubled-angle resultant."""
    fiber = normalize_vectors(fiber_direction)
    weight = np.asarray(weights, dtype=float)
    if fiber.ndim != 2 or fiber.shape[1] != 2:
        raise ValueError("fiber_direction must have shape (families, 2).")
    if weight.shape != (fiber.shape[0],):
        raise ValueError("weights must have shape (families,).")
    if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
        raise ValueError("weights must be finite and non-negative.")
    if np.sum(weight) <= 0.0:
        raise ValueError("weights must contain a positive value.")
    weight = weight / np.sum(weight)
    angle = vector_to_angle(fiber)
    cosine = float(np.sum(weight * np.cos(2.0 * angle)))
    sine = float(np.sum(weight * np.sin(2.0 * angle)))
    return float(np.hypot(cosine, sine))


def element_adjacency_edges(mesh: StructuredQuadMesh) -> IntArray:
    """Return unique horizontal and vertical element-neighbor pairs."""
    nx = mesh.number_of_elements_x
    ny = mesh.number_of_elements_y
    edges: list[tuple[int, int]] = []
    for row in range(ny):
        for column in range(nx):
            element = row * nx + column
            if column + 1 < nx:
                edges.append((element, element + 1))
            if row + 1 < ny:
                edges.append((element, element + nx))
    return np.asarray(edges, dtype=np.int64)


def element_graph_laplacian(mesh: StructuredQuadMesh) -> csr_matrix:
    """Return the unweighted combinatorial Laplacian of the element graph."""
    edges = element_adjacency_edges(mesh)
    size = mesh.number_of_elements
    if edges.size == 0:
        return csr_matrix((size, size), dtype=float)

    row = np.concatenate((edges[:, 0], edges[:, 1]))
    column = np.concatenate((edges[:, 1], edges[:, 0]))
    data = -np.ones(row.size, dtype=float)
    off_diagonal = csr_matrix((data, (row, column)), shape=(size, size))
    degree = -np.asarray(off_diagonal.sum(axis=1)).ravel()
    return off_diagonal + csr_matrix(
        (degree, (np.arange(size), np.arange(size))),
        shape=(size, size),
    )


def regularize_scalar_field(
    mesh: StructuredQuadMesh,
    values: ArrayLike,
    *,
    strength: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> FloatArray:
    """Apply one implicit graph-Laplacian regularization step.

    It solves ``(I + strength*L) x = y`` independently for each trailing field
    component. ``strength=0`` returns an exact copy.
    """
    value = np.asarray(values, dtype=float)
    if value.shape[0] != mesh.number_of_elements:
        raise ValueError("The leading field dimension must equal the number of elements.")
    if not np.all(np.isfinite(value)):
        raise ValueError("values must be finite.")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and non-negative.")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise ValueError("lower_bound cannot exceed upper_bound.")

    if strength == 0.0:
        result = value.copy()
    else:
        laplacian = element_graph_laplacian(mesh)
        operator = eye(mesh.number_of_elements, format="csr") + strength * laplacian
        flat = value.reshape(mesh.number_of_elements, -1)
        regularized = np.column_stack(
            [spsolve(operator, flat[:, column]) for column in range(flat.shape[1])]
        )
        result = regularized.reshape(value.shape)

    if lower_bound is not None:
        result = np.maximum(result, lower_bound)
    if upper_bound is not None:
        result = np.minimum(result, upper_bound)
    return np.asarray(result, dtype=float)


def regularize_nematic_field(
    mesh: StructuredQuadMesh,
    fiber_direction: ArrayLike,
    *,
    strength: float,
) -> FloatArray:
    """Regularize undirected directions in doubled-angle space."""
    fiber = np.asarray(fiber_direction, dtype=float)
    if fiber.shape[0] != mesh.number_of_elements or fiber.shape[-1] != 2:
        raise ValueError(
            "fiber_direction must have shape (elements, ..., 2)."
        )
    fiber = normalize_vectors(fiber)
    angle = vector_to_angle(fiber)
    doubled = np.stack((np.cos(2.0 * angle), np.sin(2.0 * angle)), axis=-1)
    smoothed = regularize_scalar_field(mesh, doubled, strength=strength)
    magnitude = np.linalg.norm(smoothed, axis=-1)
    near_zero = magnitude < 1.0e-12
    if np.any(near_zero):
        smoothed[near_zero] = doubled[near_zero]
    smoothed /= np.linalg.norm(smoothed, axis=-1, keepdims=True)
    regularized_angle = 0.5 * np.arctan2(smoothed[..., 1], smoothed[..., 0])
    return angle_to_vector(regularized_angle)


def scalar_graph_roughness(
    mesh: StructuredQuadMesh,
    values: ArrayLike,
) -> float:
    """Return mean squared jump across adjacent elements."""
    value = np.asarray(values, dtype=float)
    if value.shape[0] != mesh.number_of_elements:
        raise ValueError("The leading field dimension must equal the number of elements.")
    edges = element_adjacency_edges(mesh)
    if edges.size == 0:
        return 0.0
    difference = value[edges[:, 0]] - value[edges[:, 1]]
    return float(np.mean(difference**2))


def nematic_graph_roughness(
    mesh: StructuredQuadMesh,
    fiber_direction: ArrayLike,
) -> float:
    """Return mean ``1-(a_i dot a_j)^2`` over adjacent elements."""
    fiber = normalize_vectors(fiber_direction)
    if fiber.shape != (mesh.number_of_elements, 2):
        raise ValueError("fiber_direction must have shape (elements, 2).")
    edges = element_adjacency_edges(mesh)
    if edges.size == 0:
        return 0.0
    alignment = np.einsum(
        "ei,ei->e",
        fiber[edges[:, 0]],
        fiber[edges[:, 1]],
    )
    return float(np.mean(1.0 - alignment**2))
