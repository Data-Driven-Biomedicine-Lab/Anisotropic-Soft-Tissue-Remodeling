"""Finite-strain constitutive equations for a matrix reinforced by one fiber family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .orientation import normalize_vectors, orientation_tensor

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MaterialParameters:
    """Material parameters for the minimal compressible anisotropic model.

    Units are consistent but otherwise user-defined. If stresses are expressed
    in kPa, ``mu``, ``kappa`` and ``k1`` must all be given in kPa.
    """

    mu: float = 10.0
    kappa: float = 1000.0
    k1: float = 2.0
    k2: float = 5.0

    def __post_init__(self) -> None:
        for name in ("mu", "kappa", "k1", "k2"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive.")


def _validate_inputs(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("The deformation gradient must have shape (..., 2, 2).")
    if not np.all(np.isfinite(deformation)):
        raise ValueError("The deformation gradient contains non-finite values.")

    determinant = np.linalg.det(deformation)
    if np.any(determinant <= 0.0):
        raise ValueError("The deformation gradient must satisfy det(F) > 0.")

    fiber = normalize_vectors(fiber_direction)
    order = np.asarray(structural_order, dtype=float)
    if not np.all(np.isfinite(order)):
        raise ValueError("The structural-order parameter contains non-finite values.")
    if np.any((order < 0.0) | (order > 1.0)):
        raise ValueError("The structural-order parameter beta must lie in [0, 1].")

    batch_shape = np.broadcast_shapes(
        deformation.shape[:-2],
        fiber.shape[:-1],
        order.shape,
    )
    deformation = np.broadcast_to(deformation, batch_shape + (2, 2))
    fiber = np.broadcast_to(fiber, batch_shape + (2,))
    order = np.broadcast_to(order, batch_shape)
    determinant = np.broadcast_to(determinant, batch_shape)
    return deformation, fiber, order, determinant


def invariants(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute J, I1 = tr(C), and I4 = a0 \u00b7 C a0."""
    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape[-2:] != (2, 2):
        raise ValueError("The deformation gradient must have shape (..., 2, 2).")
    determinant = np.linalg.det(deformation)
    if np.any(determinant <= 0.0):
        raise ValueError("The deformation gradient must satisfy det(F) > 0.")

    fiber = normalize_vectors(fiber_direction)
    batch_shape = np.broadcast_shapes(deformation.shape[:-2], fiber.shape[:-1])
    deformation = np.broadcast_to(deformation, batch_shape + (2, 2))
    fiber = np.broadcast_to(fiber, batch_shape + (2,))
    determinant = np.broadcast_to(determinant, batch_shape)

    right_cauchy_green = np.einsum("...ki,...kj->...ij", deformation, deformation)
    first_invariant = np.trace(right_cauchy_green, axis1=-2, axis2=-1)
    fourth_invariant = np.einsum(
        "...i,...ij,...j->...",
        fiber,
        right_cauchy_green,
        fiber,
    )
    return determinant, first_invariant, fourth_invariant


def strain_energy_density(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MaterialParameters = MaterialParameters(),
) -> FloatArray:
    """Return total strain-energy density.

    The matrix is compressible neo-Hookean. The fiber family contributes only
    in extension through a tension-only exponential term. ``beta`` scales the
    fiber contribution and represents structural order.
    """
    deformation, fiber, order, determinant = _validate_inputs(
        deformation_gradient,
        fiber_direction,
        structural_order,
    )
    _, first_invariant, fourth_invariant = invariants(deformation, fiber)
    logarithmic_jacobian = np.log(determinant)

    matrix_energy = (
        0.5 * parameters.mu * (first_invariant - 2.0 - 2.0 * logarithmic_jacobian)
        + 0.5 * parameters.kappa * logarithmic_jacobian**2
    )

    fiber_extension = np.maximum(fourth_invariant - 1.0, 0.0)
    fiber_energy = (
        order
        * parameters.k1
        / (2.0 * parameters.k2)
        * np.expm1(parameters.k2 * fiber_extension**2)
    )
    return matrix_energy + fiber_energy


def first_piola_stress(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MaterialParameters = MaterialParameters(),
) -> FloatArray:
    """Return the first Piola-Kirchhoff stress P = d psi / d F."""
    deformation, fiber, order, determinant = _validate_inputs(
        deformation_gradient,
        fiber_direction,
        structural_order,
    )
    inverse_transpose = np.swapaxes(np.linalg.inv(deformation), -1, -2)
    logarithmic_jacobian = np.log(determinant)

    matrix_stress = (
        parameters.mu * (deformation - inverse_transpose)
        + parameters.kappa * logarithmic_jacobian[..., None, None] * inverse_transpose
    )

    structural_tensor = orientation_tensor(fiber)
    right_cauchy_green = np.einsum("...ki,...kj->...ij", deformation, deformation)
    fourth_invariant = np.einsum(
        "...i,...ij,...j->...",
        fiber,
        right_cauchy_green,
        fiber,
    )
    fiber_extension = np.maximum(fourth_invariant - 1.0, 0.0)
    fiber_multiplier = (
        2.0
        * order
        * parameters.k1
        * fiber_extension
        * np.exp(parameters.k2 * fiber_extension**2)
    )
    deformation_times_structure = np.einsum(
        "...ik,...kj->...ij",
        deformation,
        structural_tensor,
    )
    fiber_stress = fiber_multiplier[..., None, None] * deformation_times_structure
    return matrix_stress + fiber_stress


def cauchy_stress(
    deformation_gradient: ArrayLike,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameters: MaterialParameters = MaterialParameters(),
) -> FloatArray:
    """Return the symmetric Cauchy stress sigma = J^(-1) P F^T."""
    deformation, fiber, order, determinant = _validate_inputs(
        deformation_gradient,
        fiber_direction,
        structural_order,
    )
    piola = first_piola_stress(deformation, fiber, order, parameters)
    return np.einsum(
        "...ik,...jk->...ij",
        piola,
        deformation,
    ) / determinant[..., None, None]
