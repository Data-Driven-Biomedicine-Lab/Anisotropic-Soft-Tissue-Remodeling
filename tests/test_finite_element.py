import numpy as np

from anisotropic_remodeling.finite_element import (
    FiniteElementConfig,
    assemble_internal_energy_and_force,
    element_centroids,
    rectangular_quad_mesh,
    sample_nematic_image_to_elements,
    solve_displacement_controlled_equilibrium,
)
from anisotropic_remodeling.material import MaterialParameters, strain_energy_density


def test_affine_field_is_integrated_exactly_for_constant_state() -> None:
    mesh = rectangular_quad_mesh(3, 2, width=2.0, height=1.0)
    deformation = np.array([[1.08, 0.07], [0.02, 0.96]])
    displacement = (mesh.nodes @ (deformation - np.eye(2)).T)
    fiber = np.tile(np.array([0.8, 0.6]), (mesh.number_of_elements, 1))
    beta = np.full(mesh.number_of_elements, 0.55)
    material = MaterialParameters(mu=2.0, kappa=80.0, k1=3.0, k2=4.0)

    energy, _, element_f, element_j, element_energy, _ = assemble_internal_energy_and_force(
        mesh, displacement, fiber, beta, material
    )
    expected_density = float(strain_energy_density(deformation, fiber[0], beta[0], material))

    np.testing.assert_allclose(
        element_f, np.broadcast_to(deformation, element_f.shape), atol=1e-12
    )
    np.testing.assert_allclose(element_j, np.linalg.det(deformation), atol=1e-12)
    np.testing.assert_allclose(element_energy, expected_density, atol=1e-12)
    np.testing.assert_allclose(energy, expected_density * mesh.width * mesh.height, atol=1e-12)


def test_equilibrium_solver_balances_reactions_and_free_residual() -> None:
    mesh = rectangular_quad_mesh(4, 2, width=2.0, height=1.0)
    fiber = np.tile(np.array([1.0, 0.0]), (mesh.number_of_elements, 1))
    beta = np.full(mesh.number_of_elements, 0.4)
    result = solve_displacement_controlled_equilibrium(
        mesh,
        fiber,
        beta,
        MaterialParameters(mu=2.0, kappa=60.0, k1=1.0, k2=2.0),
        FiniteElementConfig(
            axial_extension=0.05,
            load_steps=3,
            gradient_tolerance=2e-7,
            maximum_iterations=400,
        ),
    )

    assert np.all(result.converged)
    assert result.free_dof_residual_norm < 3e-6
    assert np.min(result.element_jacobian) > 0.2
    np.testing.assert_allclose(result.left_reaction + result.right_reaction, 0.0, atol=2e-5)


def test_nematic_sampling_is_invariant_to_pi_flips() -> None:
    x_axis = np.linspace(0.0, 1.0, 5)
    y_axis = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(x_axis, y_axis)
    angle = 0.2 + 0.5 * x
    fiber = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    beta = 0.2 + 0.6 * y
    mask = np.ones_like(x, dtype=bool)
    points = element_centroids(rectangular_quad_mesh(3, 2, width=1.0, height=1.0))

    sampled_a, sampled_beta = sample_nematic_image_to_elements(
        points, x, y, fiber, beta, mask, neighbors=6
    )
    flipped = fiber.copy()
    flipped[::2, ::2] *= -1.0
    sampled_flipped, sampled_beta_flipped = sample_nematic_image_to_elements(
        points, x, y, flipped, beta, mask, neighbors=6
    )

    tensor = np.einsum("...i,...j->...ij", sampled_a, sampled_a)
    tensor_flipped = np.einsum("...i,...j->...ij", sampled_flipped, sampled_flipped)
    np.testing.assert_allclose(tensor, tensor_flipped, atol=1e-12)
    np.testing.assert_allclose(sampled_beta, sampled_beta_flipped, atol=1e-12)


def test_equilibrium_solver_accepts_a_warm_start() -> None:
    mesh = rectangular_quad_mesh(2, 1, width=2.0, height=1.0)
    fiber = np.tile(np.array([1.0, 0.0]), (mesh.number_of_elements, 1))
    beta = np.full(mesh.number_of_elements, 0.4)
    material = MaterialParameters(mu=2.0, kappa=60.0, k1=1.0, k2=2.0)
    config = FiniteElementConfig(
        axial_extension=0.04,
        load_steps=2,
        gradient_tolerance=5e-7,
        maximum_iterations=300,
    )
    first = solve_displacement_controlled_equilibrium(mesh, fiber, beta, material, config)
    second = solve_displacement_controlled_equilibrium(
        mesh,
        fiber,
        beta,
        material,
        FiniteElementConfig(
            axial_extension=0.04,
            load_steps=1,
            gradient_tolerance=5e-7,
            maximum_iterations=300,
        ),
        initial_displacement=first.displacement,
    )

    np.testing.assert_allclose(second.displacement, first.displacement, atol=2e-6)
    assert second.free_dof_residual_norm < 5e-6
