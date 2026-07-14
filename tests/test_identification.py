import numpy as np

from anisotropic_remodeling.architecture import MultiFiberMaterialParameters
from anisotropic_remodeling.identification import (
    MaterialParameterMap,
    build_multiaxial_protocol_dataset,
    fit_material_parameters,
    local_sensitivity_matrix,
    parametric_bootstrap_material_fit,
    predict_dataset_stress,
)
from anisotropic_remodeling.orientation import angle_to_vector


def benchmark_structure():
    fiber = angle_to_vector(np.deg2rad([18.0, 102.0]))
    beta = np.array([0.78, 0.46])
    material = MultiFiberMaterialParameters(
        mu=2.4,
        kappa=180.0,
        k1=(3.2, 1.7),
        k2=(4.6, 3.4),
        family_weights=(0.65, 0.35),
    )
    return fiber, beta, material


def test_parameter_map_round_trip() -> None:
    _, _, material = benchmark_structure()
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=material.family_weights,
        identify_kappa=True,
    )
    reconstructed = mapping.unpack(mapping.pack(material))
    assert reconstructed == material


def test_dataset_generation_is_reproducible() -> None:
    fiber, beta, material = benchmark_structure()
    first = build_multiaxial_protocol_dataset(
        material,
        fiber,
        beta,
        dilation_values=np.linspace(1.0, 1.015, 7),
        random_seed=9,
    )
    second = build_multiaxial_protocol_dataset(
        material,
        fiber,
        beta,
        dilation_values=np.linspace(1.0, 1.015, 7),
        random_seed=9,
    )
    np.testing.assert_allclose(first.observed_stress, second.observed_stress)
    assert first.protocols == (
        "uniaxial_x",
        "uniaxial_y",
        "simple_shear",
        "dilation",
    )


def test_noiseless_fit_recovers_parameters() -> None:
    fiber, beta, material = benchmark_structure()
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=material.family_weights,
        identify_kappa=True,
    )
    noisy = build_multiaxial_protocol_dataset(
        material,
        fiber,
        beta,
        axial_stretches=np.linspace(1.0, 1.18, 15),
        shear_values=np.linspace(0.0, 0.22, 13),
        dilation_values=np.linspace(1.0, 1.018, 9),
        relative_noise=0.0,
        absolute_noise=1.0e-4,
        random_seed=1,
    )
    clean = noisy.with_observed_stress(
        predict_dataset_stress(noisy, fiber, beta, material)
    )
    true_values = mapping.pack(material)
    fit = fit_material_parameters(
        clean,
        fiber,
        beta,
        mapping,
        initial_values=true_values * np.array([0.8, 0.75, 1.2, 1.1, 0.8, 0.7]),
        lower_bounds=true_values * 0.2,
        upper_bounds=true_values * 5.0,
    )
    np.testing.assert_allclose(
        fit.parameter_vector,
        true_values,
        rtol=2.0e-4,
        atol=2.0e-5,
    )
    assert fit.success


def test_kappa_is_unobservable_in_isochoric_protocols() -> None:
    fiber, beta, material = benchmark_structure()
    dataset = build_multiaxial_protocol_dataset(
        material,
        fiber,
        beta,
        dilation_values=None,
        relative_noise=0.0,
        absolute_noise=0.01,
        random_seed=2,
    )
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=material.family_weights,
        identify_kappa=True,
    )
    sensitivity = local_sensitivity_matrix(
        dataset,
        fiber,
        beta,
        material,
        mapping,
    )
    np.testing.assert_allclose(sensitivity[:, -1], 0.0, atol=1.0e-10)


def test_dilation_restores_kappa_sensitivity() -> None:
    fiber, beta, material = benchmark_structure()
    dataset = build_multiaxial_protocol_dataset(
        material,
        fiber,
        beta,
        dilation_values=np.linspace(1.0, 1.015, 6),
        relative_noise=0.0,
        absolute_noise=0.01,
        random_seed=2,
    )
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=material.family_weights,
        identify_kappa=True,
    )
    sensitivity = local_sensitivity_matrix(
        dataset,
        fiber,
        beta,
        material,
        mapping,
    )
    assert np.linalg.norm(sensitivity[:, -1]) > 1.0e-3


def test_bootstrap_is_reproducible() -> None:
    fiber, beta, material = benchmark_structure()
    dataset = build_multiaxial_protocol_dataset(
        material,
        fiber,
        beta,
        axial_stretches=np.linspace(1.0, 1.15, 9),
        shear_values=np.linspace(0.0, 0.15, 7),
        dilation_values=np.linspace(1.0, 1.012, 5),
        random_seed=7,
    )
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=material.family_weights,
        identify_kappa=True,
    )
    true_values = mapping.pack(material)
    fit = fit_material_parameters(
        dataset,
        fiber,
        beta,
        mapping,
        initial_values=true_values * 0.9,
        lower_bounds=true_values * 0.2,
        upper_bounds=true_values * 5.0,
    )
    first = parametric_bootstrap_material_fit(
        dataset,
        fiber,
        beta,
        mapping,
        fit,
        lower_bounds=true_values * 0.2,
        upper_bounds=true_values * 5.0,
        number_of_samples=3,
        random_seed=11,
    )
    second = parametric_bootstrap_material_fit(
        dataset,
        fiber,
        beta,
        mapping,
        fit,
        lower_bounds=true_values * 0.2,
        upper_bounds=true_values * 5.0,
        number_of_samples=3,
        random_seed=11,
    )
    np.testing.assert_allclose(
        first.parameter_samples,
        second.parameter_samples,
        equal_nan=True,
    )
    np.testing.assert_array_equal(first.successful, second.successful)
