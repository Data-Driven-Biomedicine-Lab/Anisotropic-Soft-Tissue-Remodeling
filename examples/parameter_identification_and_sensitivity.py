"""Run the version-0.8 parameter-identification benchmark."""

from __future__ import annotations

import numpy as np

from anisotropic_remodeling import (
    MaterialParameterMap,
    MultiFiberMaterialParameters,
    angle_to_vector,
    build_multiaxial_protocol_dataset,
    fit_material_parameters,
)


def main() -> None:
    fiber = angle_to_vector(np.deg2rad([18.0, 102.0]))
    beta = np.array([0.78, 0.46])
    truth = MultiFiberMaterialParameters(
        mu=2.4,
        kappa=180.0,
        k1=(3.2, 1.7),
        k2=(4.6, 3.4),
        family_weights=(0.65, 0.35),
    )
    dataset = build_multiaxial_protocol_dataset(
        truth,
        fiber,
        beta,
        dilation_values=np.linspace(1.0, 1.018, 9),
        random_seed=2026,
    )
    mapping = MaterialParameterMap(
        number_of_families=2,
        family_weights=truth.family_weights,
        identify_kappa=True,
    )
    true_values = mapping.pack(truth)
    fit = fit_material_parameters(
        dataset,
        fiber,
        beta,
        mapping,
        initial_values=np.array([1.8, 2.2, 2.2, 3.8, 3.8, 120.0]),
        lower_bounds=np.array([0.2, 0.2, 0.2, 0.5, 0.5, 20.0]),
        upper_bounds=np.array([8.0, 10.0, 10.0, 12.0, 12.0, 600.0]),
    )

    print(f"{'parameter':>12} {'truth':>12} {'estimate':>12} {'std. error':>12}")
    for name, true, estimate, standard_error in zip(
        fit.parameter_names,
        true_values,
        fit.parameter_vector,
        fit.standard_error,
        strict=True,
    ):
        print(f"{name:>12} {true:12.6f} {estimate:12.6f} {standard_error:12.6f}")
    print(f"Objective: {fit.objective:.6f}")
    print(f"Condition number: {fit.condition_number:.3e}")


if __name__ == "__main__":
    main()
