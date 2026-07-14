"""Run the fully synthetic blind validation challenge."""

from __future__ import annotations

import numpy as np

from anisotropic_remodeling import (
    MaterialParameterMap,
    create_synthetic_validation_challenge,
    evaluate_synthetic_challenge,
    fit_material_parameters,
    predict_dataset_stress,
)


def main() -> None:
    challenge = create_synthetic_validation_challenge()
    public = challenge.public
    parameter_map = MaterialParameterMap(
        number_of_families=2,
        family_weights=public.family_weights,
        identify_kappa=True,
    )
    fit = fit_material_parameters(
        public.training_dataset,
        public.fiber_direction,
        public.structural_order,
        parameter_map,
        initial_values=np.array([2.0, 2.6, 2.0, 4.0, 3.8, 150.0]),
        lower_bounds=np.array([0.2, 0.2, 0.2, 0.5, 0.5, 20.0]),
        upper_bounds=np.array([8.0, 12.0, 12.0, 14.0, 14.0, 700.0]),
    )
    prediction = predict_dataset_stress(
        public.empty_test_dataset(),
        public.fiber_direction,
        public.structural_order,
        fit.material,
    )
    evaluation = evaluate_synthetic_challenge(challenge, prediction)

    print(f"Challenge: {public.challenge_id}")
    print(f"Training protocols: {public.training_protocols}")
    print(f"Test protocols: {public.test_protocols}")
    print(f"Fit success: {fit.success}")
    print(f"Held-out RMSE: {evaluation.overall.rmse:.6f}")
    print(f"Held-out normalized RMSE: {evaluation.overall.normalized_rmse:.3%}")
    print(f"Held-out R^2: {evaluation.overall.r_squared:.6f}")


if __name__ == "__main__":
    main()
