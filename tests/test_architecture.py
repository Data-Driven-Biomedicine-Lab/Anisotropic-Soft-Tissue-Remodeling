import numpy as np

from anisotropic_remodeling.architecture import (
    MultiFiberMaterialParameters,
    discrete_nematic_coherence,
    discrete_nematic_distribution,
    multifiber_cauchy_stress,
    multifiber_first_piola_stress,
    multifiber_strain_energy_density,
    nematic_graph_roughness,
    regularize_nematic_field,
    regularize_scalar_field,
    scalar_graph_roughness,
    theoretical_nematic_coherence,
)
from anisotropic_remodeling.finite_element import rectangular_quad_mesh
from anisotropic_remodeling.material import (
    MaterialParameters,
    first_piola_stress,
    strain_energy_density,
)
from anisotropic_remodeling.orientation import angle_to_vector


def test_single_family_reduces_to_original_material() -> None:
    deformation = np.array([[1.10, 0.06], [0.02, 0.95]])
    fiber = np.array([0.8, 0.6])
    beta = 0.65
    original = MaterialParameters(mu=2.0, kappa=80.0, k1=3.0, k2=4.0)
    multi = MultiFiberMaterialParameters(
        mu=2.0,
        kappa=80.0,
        k1=(3.0,),
        k2=(4.0,),
        family_weights=(1.0,),
    )

    np.testing.assert_allclose(
        multifiber_strain_energy_density(
            deformation,
            fiber[None, :],
            np.array([beta]),
            multi,
        ),
        strain_energy_density(deformation, fiber, beta, original),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        multifiber_first_piola_stress(
            deformation,
            fiber[None, :],
            np.array([beta]),
            multi,
        ),
        first_piola_stress(deformation, fiber, beta, original),
        atol=1e-12,
    )


def test_multifiber_piola_matches_energy_gradient() -> None:
    deformation = np.array([[1.12, 0.08], [0.03, 0.94]])
    directions = angle_to_vector(np.deg2rad([20.0, 105.0]))
    beta = np.array([0.7, 0.4])
    parameters = MultiFiberMaterialParameters(
        mu=2.0,
        kappa=90.0,
        k1=(3.0, 1.5),
        k2=(4.0, 3.0),
        family_weights=(0.65, 0.35),
    )
    analytical = multifiber_first_piola_stress(
        deformation,
        directions,
        beta,
        parameters,
    )
    epsilon = 1e-6
    numerical = np.empty((2, 2))
    for row in range(2):
        for column in range(2):
            perturbation = np.zeros((2, 2))
            perturbation[row, column] = epsilon
            plus = multifiber_strain_energy_density(
                deformation + perturbation,
                directions,
                beta,
                parameters,
            )
            minus = multifiber_strain_energy_density(
                deformation - perturbation,
                directions,
                beta,
                parameters,
            )
            numerical[row, column] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-7)


def test_nematic_distribution_matches_theoretical_coherence() -> None:
    _, directions, weights = discrete_nematic_distribution(
        np.deg2rad(30.0),
        3.0,
        number_of_directions=720,
    )
    discrete = discrete_nematic_coherence(directions, weights)
    theoretical = theoretical_nematic_coherence(3.0)
    np.testing.assert_allclose(discrete, theoretical, rtol=2e-5, atol=2e-5)


def test_multifiber_response_is_invariant_to_direction_flips() -> None:
    deformation = np.array([[1.08, 0.04], [0.01, 0.97]])
    directions = angle_to_vector(np.deg2rad([15.0, 95.0]))
    beta = np.array([0.7, 0.5])
    parameters = MultiFiberMaterialParameters(
        mu=2.0,
        kappa=80.0,
        k1=(2.0, 1.0),
        k2=(4.0, 3.0),
        family_weights=(0.7, 0.3),
    )
    flipped = directions.copy()
    flipped[1] *= -1.0
    np.testing.assert_allclose(
        multifiber_cauchy_stress(deformation, directions, beta, parameters),
        multifiber_cauchy_stress(deformation, flipped, beta, parameters),
        atol=1e-12,
    )


def test_graph_regularization_reduces_roughness_and_preserves_bounds() -> None:
    mesh = rectangular_quad_mesh(5, 3)
    rng = np.random.default_rng(4)
    beta = np.clip(0.5 + 0.2 * rng.normal(size=mesh.number_of_elements), 0.0, 1.0)
    angle = 0.4 + 0.35 * rng.normal(size=mesh.number_of_elements)
    fiber = angle_to_vector(angle)

    regularized_beta = regularize_scalar_field(
        mesh,
        beta,
        strength=0.8,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    regularized_fiber = regularize_nematic_field(mesh, fiber, strength=0.8)

    assert scalar_graph_roughness(mesh, regularized_beta) < scalar_graph_roughness(
        mesh, beta
    )
    assert nematic_graph_roughness(
        mesh, regularized_fiber
    ) < nematic_graph_roughness(mesh, fiber)
    assert np.all((regularized_beta >= 0.0) & (regularized_beta <= 1.0))
    np.testing.assert_allclose(
        np.linalg.norm(regularized_fiber, axis=-1),
        1.0,
        atol=1e-12,
    )
