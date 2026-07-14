import numpy as np
import pytest

from anisotropic_remodeling.material import (
    MaterialParameters,
    cauchy_stress,
    first_piola_stress,
    strain_energy_density,
)

PARAMETERS = MaterialParameters(mu=2.0, kappa=100.0, k1=4.0, k2=3.0)
FIBER = np.array([1.0, 0.0])


def test_reference_state_is_stress_and_energy_free() -> None:
    deformation = np.eye(2)
    np.testing.assert_allclose(
        strain_energy_density(deformation, FIBER, 0.7, PARAMETERS),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first_piola_stress(deformation, FIBER, 0.7, PARAMETERS),
        np.zeros((2, 2)),
        atol=1e-12,
    )


def test_fiber_is_inactive_in_compression() -> None:
    deformation = np.diag([0.9, 1.0 / 0.9])
    energy_ordered = strain_energy_density(deformation, FIBER, 1.0, PARAMETERS)
    energy_disordered = strain_energy_density(deformation, FIBER, 0.0, PARAMETERS)
    np.testing.assert_allclose(energy_ordered, energy_disordered, atol=1e-12)


def test_cauchy_stress_is_symmetric() -> None:
    deformation = np.array([[1.2, 0.15], [0.0, 0.9]])
    stress = cauchy_stress(deformation, np.array([0.8, 0.6]), 0.75, PARAMETERS)
    np.testing.assert_allclose(stress, stress.T, atol=1e-12)


def test_piola_stress_matches_energy_gradient() -> None:
    deformation = np.array([[1.12, 0.08], [0.03, 0.94]])
    fiber = np.array([0.8, 0.6])
    beta = 0.65
    analytical = first_piola_stress(deformation, fiber, beta, PARAMETERS)

    epsilon = 1e-6
    numerical = np.empty((2, 2))
    for row in range(2):
        for column in range(2):
            perturbation = np.zeros((2, 2))
            perturbation[row, column] = epsilon
            energy_plus = strain_energy_density(
                deformation + perturbation, fiber, beta, PARAMETERS
            )
            energy_minus = strain_energy_density(
                deformation - perturbation, fiber, beta, PARAMETERS
            )
            numerical[row, column] = (energy_plus - energy_minus) / (2.0 * epsilon)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-7)


def test_invalid_jacobian_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"det\(F\) > 0"):
        strain_energy_density(np.diag([-1.0, 1.0]), FIBER, 0.5, PARAMETERS)
