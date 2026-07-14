import numpy as np

from anisotropic_remodeling.stimuli import (
    directional_stretch_stimulus,
    equilibrium_structural_order,
    hill_activation,
    principal_stretch_direction,
)


def test_principal_direction_for_uniaxial_stretch() -> None:
    deformation = np.diag([1.3, 1.0 / 1.3])
    direction = principal_stretch_direction(deformation)
    np.testing.assert_allclose(direction, np.array([1.0, 0.0]), atol=1e-12)


def test_directional_stimulus_vanishes_for_isotropic_stretch() -> None:
    deformation = 1.2 * np.eye(2)
    np.testing.assert_allclose(directional_stretch_stimulus(deformation), 0.0, atol=1e-12)


def test_equilibrium_order_is_bounded_and_monotone() -> None:
    stimulus = np.linspace(0.0, 2.0, 101)
    beta = equilibrium_structural_order(stimulus, beta_min=0.2, beta_max=0.9)
    assert np.all(np.diff(beta) >= 0.0)
    assert np.all((beta >= 0.2) & (beta <= 0.9))


def test_hill_activation_is_zero_without_directional_stimulus() -> None:
    np.testing.assert_allclose(hill_activation(0.0), 0.0, atol=1e-12)
