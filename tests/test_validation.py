import numpy as np

from anisotropic_remodeling import (
    MaterialParameterMap,
    evaluate_synthetic_challenge,
    fit_material_parameters,
    predict_dataset_stress,
)
from anisotropic_remodeling.validation import (
    create_synthetic_validation_challenge,
    perturb_synthetic_architecture,
    validation_metrics,
)


def test_challenge_generation_is_reproducible() -> None:
    first = create_synthetic_validation_challenge(random_seed=5)
    second = create_synthetic_validation_challenge(random_seed=5)
    np.testing.assert_allclose(
        first.public.training_dataset.observed_stress,
        second.public.training_dataset.observed_stress,
    )
    np.testing.assert_allclose(
        first.hidden_test_stress_observed,
        second.hidden_test_stress_observed,
    )
    assert first.metadata["data_origin"] == "fully synthetic"


def test_training_and_test_protocols_are_disjoint() -> None:
    challenge = create_synthetic_validation_challenge()
    assert set(challenge.public.training_protocols).isdisjoint(
        challenge.public.test_protocols
    )


def test_perfect_prediction_has_perfect_scores() -> None:
    observed = np.array([1.0, 2.0, 3.0])
    metric = validation_metrics(observed, observed, np.ones(3))
    assert metric.rmse == 0.0
    assert metric.mae == 0.0
    assert metric.normalized_rmse == 0.0
    assert metric.r_squared == 1.0
    assert metric.fraction_within_two_sigma == 1.0


def test_hidden_truth_predicts_clean_test_exactly() -> None:
    challenge = create_synthetic_validation_challenge()
    prediction = predict_dataset_stress(
        challenge.public.empty_test_dataset(),
        challenge.public.fiber_direction,
        challenge.public.structural_order,
        challenge.hidden_material,
    )
    evaluation = evaluate_synthetic_challenge(
        challenge,
        prediction,
        compare_to_clean_truth=True,
    )
    assert evaluation.overall.rmse < 1.0e-12


def test_noiseless_blind_fit_predicts_held_out_protocols() -> None:
    challenge = create_synthetic_validation_challenge(
        random_seed=8,
        relative_noise=0.0,
        absolute_noise=1.0e-5,
    )
    training_clean = predict_dataset_stress(
        challenge.public.training_dataset,
        challenge.public.fiber_direction,
        challenge.public.structural_order,
        challenge.hidden_material,
    )
    training_dataset = challenge.public.training_dataset.with_observed_stress(
        training_clean
    )
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=challenge.public.family_weights,
        identify_kappa=True,
    )
    truth = mapping.pack(challenge.hidden_material)
    fit = fit_material_parameters(
        training_dataset,
        challenge.public.fiber_direction,
        challenge.public.structural_order,
        mapping,
        initial_values=truth * np.array([0.8, 0.8, 1.2, 0.9, 1.1, 0.75]),
        lower_bounds=truth * 0.2,
        upper_bounds=truth * 5.0,
    )
    prediction = predict_dataset_stress(
        challenge.public.empty_test_dataset(),
        challenge.public.fiber_direction,
        challenge.public.structural_order,
        fit.material,
    )
    evaluation = evaluate_synthetic_challenge(
        challenge,
        prediction,
        compare_to_clean_truth=True,
    )
    assert fit.success
    assert evaluation.overall.normalized_rmse < 1.0e-4


def test_architecture_perturbation_is_reproducible_and_bounded() -> None:
    challenge = create_synthetic_validation_challenge()
    first = perturb_synthetic_architecture(
        challenge.public.fiber_direction,
        challenge.public.structural_order,
        angle_noise_std_deg=4.0,
        order_noise_std=0.05,
        random_seed=11,
    )
    second = perturb_synthetic_architecture(
        challenge.public.fiber_direction,
        challenge.public.structural_order,
        angle_noise_std_deg=4.0,
        order_noise_std=0.05,
        random_seed=11,
    )
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    np.testing.assert_allclose(np.linalg.norm(first[0], axis=-1), 1.0)
    assert np.all((first[1] >= 0.0) & (first[1] <= 1.0))
