import numpy as np

from anisotropic_remodeling.architecture import MultiFiberMaterialParameters
from anisotropic_remodeling.finite_element import FiniteElementConfig, rectangular_quad_mesh
from anisotropic_remodeling.multifiber_finite_element import (
    assemble_multifiber_internal_energy_and_force,
    solve_multifiber_displacement_controlled_equilibrium,
)
from anisotropic_remodeling.orientation import angle_to_vector


def test_affine_multifiber_field_is_integrated_exactly() -> None:
    mesh = rectangular_quad_mesh(2, 2, width=2.0, height=1.0)
    deformation = np.array([[1.06, 0.04], [0.01, 0.97]])
    displacement = mesh.nodes @ (deformation - np.eye(2)).T
    families = angle_to_vector(np.deg2rad([10.0, 100.0]))
    fiber = np.broadcast_to(
        families,
        (mesh.number_of_elements, 2, 2),
    ).copy()
    beta = np.broadcast_to(
        np.array([0.6, 0.35]),
        (mesh.number_of_elements, 2),
    ).copy()
    material = MultiFiberMaterialParameters(
        mu=2.0,
        kappa=70.0,
        k1=(2.0, 1.0),
        k2=(3.0, 2.5),
        family_weights=(0.65, 0.35),
    )

    _, _, element_f, element_j, _, _ = (
        assemble_multifiber_internal_energy_and_force(
            mesh,
            displacement,
            fiber,
            beta,
            material,
        )
    )
    np.testing.assert_allclose(
        element_f,
        np.broadcast_to(deformation, element_f.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(element_j, np.linalg.det(deformation), atol=1e-12)


def test_multifiber_equilibrium_balances_reactions() -> None:
    mesh = rectangular_quad_mesh(2, 1, width=2.0, height=1.0)
    families = angle_to_vector(np.deg2rad([[20.0, 100.0], [35.0, 115.0]]))
    beta = np.array([[0.6, 0.3], [0.7, 0.4]])
    material = MultiFiberMaterialParameters(
        mu=2.0,
        kappa=60.0,
        k1=(1.5, 1.0),
        k2=(3.0, 3.0),
        family_weights=(0.65, 0.35),
    )
    result = solve_multifiber_displacement_controlled_equilibrium(
        mesh,
        families,
        beta,
        material,
        FiniteElementConfig(
            axial_extension=0.04,
            load_steps=2,
            gradient_tolerance=5e-7,
            maximum_iterations=350,
        ),
    )

    assert np.all(result.converged)
    assert result.free_dof_residual_norm < 5e-6
    assert np.min(result.element_jacobian) > 0.2
    np.testing.assert_allclose(
        result.left_reaction + result.right_reaction,
        0.0,
        atol=2e-5,
    )
