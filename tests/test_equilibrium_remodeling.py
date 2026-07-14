import numpy as np

from anisotropic_remodeling.equilibrium_remodeling import (
    EquilibriumRemodelingConfig,
    run_equilibrium_remodeling,
)
from anisotropic_remodeling.finite_element import rectangular_quad_mesh
from anisotropic_remodeling.material import MaterialParameters
from anisotropic_remodeling.orientation import angle_to_vector
from anisotropic_remodeling.remodeling import RemodelingParameters


def _small_problem():
    mesh = rectangular_quad_mesh(2, 1, width=2.0, height=1.0)
    angle = np.deg2rad(np.array([55.0, 35.0]))
    fiber = angle_to_vector(angle)
    beta = np.array([0.25, 0.45])
    material = MaterialParameters(mu=2.0, kappa=60.0, k1=1.5, k2=3.0)
    return mesh, fiber, beta, material


def test_zero_rates_preserve_structure() -> None:
    mesh, fiber, beta, material = _small_problem()
    result = run_equilibrium_remodeling(
        mesh,
        fiber,
        beta,
        material,
        RemodelingParameters(
            orientation_rate=0.0,
            order_rate=0.0,
            beta_min=0.1,
            beta_max=1.0,
            half_saturation=0.15,
            hill_exponent=2.0,
        ),
        EquilibriumRemodelingConfig(
            total_time=1.0,
            dt=0.5,
            axial_extension=0.04,
            initial_load_steps=2,
            subsequent_load_steps=1,
            gradient_tolerance=5e-7,
            maximum_iterations=300,
        ),
    )

    expected_fiber = np.broadcast_to(fiber, result.fiber_direction.shape)
    expected_beta = np.broadcast_to(beta, result.structural_order.shape)
    np.testing.assert_allclose(result.fiber_direction, expected_fiber, atol=1e-12)
    np.testing.assert_allclose(result.structural_order, expected_beta, atol=1e-12)
    assert np.all(result.converged)


def test_tensile_remodeling_improves_alignment_and_preserves_bounds() -> None:
    mesh, fiber, beta, material = _small_problem()
    result = run_equilibrium_remodeling(
        mesh,
        fiber,
        beta,
        material,
        RemodelingParameters(
            orientation_rate=0.8,
            order_rate=0.6,
            beta_min=0.1,
            beta_max=1.0,
            half_saturation=0.12,
            hill_exponent=2.0,
        ),
        EquilibriumRemodelingConfig(
            total_time=2.0,
            dt=0.5,
            axial_extension=0.05,
            initial_load_steps=2,
            subsequent_load_steps=1,
            gradient_tolerance=5e-7,
            maximum_iterations=300,
        ),
    )

    assert result.mean_target_alignment[-1] > result.mean_target_alignment[0]
    assert result.mean_structural_order[-1] > result.mean_structural_order[0]
    assert np.all((result.structural_order >= 0.0) & (result.structural_order <= 1.0))
    np.testing.assert_allclose(
        np.linalg.norm(result.fiber_direction, axis=-1),
        1.0,
        atol=1e-12,
    )
    assert np.min(result.element_jacobian) > 0.2
    assert np.max(result.free_dof_residual_norm) < 1e-5


def test_time_grid_reaches_requested_final_time() -> None:
    mesh, fiber, beta, material = _small_problem()
    result = run_equilibrium_remodeling(
        mesh,
        fiber,
        beta,
        material,
        RemodelingParameters(orientation_rate=0.0, order_rate=0.0),
        EquilibriumRemodelingConfig(
            total_time=1.0,
            dt=0.3,
            axial_extension=0.03,
            initial_load_steps=2,
            subsequent_load_steps=1,
            gradient_tolerance=1e-6,
            maximum_iterations=250,
        ),
    )

    assert result.time[-1] == 1.0
    assert np.all(np.diff(result.time) > 0.0)
    assert np.max(np.diff(result.time)) <= 0.3 + 1e-12
