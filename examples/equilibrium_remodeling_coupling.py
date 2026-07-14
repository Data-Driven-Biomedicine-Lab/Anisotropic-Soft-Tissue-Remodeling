"""Run the synthetic polarimetry -> FE equilibrium -> remodeling benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anisotropic_remodeling import (
    EquilibriumRemodelingConfig,
    MaterialParameters,
    RemodelingParameters,
    element_centroids,
    polarimetry_to_structure,
    rectangular_quad_mesh,
    run_equilibrium_remodeling,
    sample_nematic_image_to_elements,
    synthetic_polarimetry_benchmark,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    figure_directory = repository_root / "results" / "figures"
    data_directory = repository_root / "results" / "data"
    figure_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)

    benchmark = synthetic_polarimetry_benchmark()
    reconstructed = polarimetry_to_structure(
        benchmark.observed_azimuth_rad,
        benchmark.observed_retardance,
        benchmark.calibration,
        minimum_valid_retardance=0.16,
        external_valid_mask=benchmark.external_valid_mask,
        coherence_window=7,
    )

    mesh = rectangular_quad_mesh(5, 3, width=2.0, height=1.0)
    fiber, beta = sample_nematic_image_to_elements(
        element_centroids(mesh),
        benchmark.x,
        benchmark.y,
        reconstructed.fiber_direction,
        reconstructed.structural_order,
        reconstructed.valid_mask,
        neighbors=20,
    )

    result = run_equilibrium_remodeling(
        mesh,
        fiber,
        beta,
        MaterialParameters(mu=2.0, kappa=60.0, k1=2.0, k2=3.0),
        RemodelingParameters(
            orientation_rate=0.7,
            order_rate=0.5,
            beta_min=0.1,
            beta_max=1.0,
            half_saturation=0.1,
            hill_exponent=2.0,
        ),
        EquilibriumRemodelingConfig(
            total_time=4.0,
            dt=0.5,
            axial_extension=0.06,
            initial_load_steps=3,
            subsequent_load_steps=1,
            gradient_tolerance=3.0e-7,
            maximum_iterations=400,
        ),
    )

    np.savez_compressed(
        data_directory / "equilibrium_remodeling_coupling_fields.npz",
        nodes=mesh.nodes,
        elements=mesh.elements,
        time=result.time,
        displacement=result.displacement,
        fiber_direction=result.fiber_direction,
        fiber_angle_deg=result.fiber_angle_deg,
        structural_order=result.structural_order,
        equilibrium_order=result.equilibrium_order,
        stimulus=result.stimulus,
        target_direction=result.target_direction,
        deformation_gradient=result.element_deformation_gradient,
        jacobian=result.element_jacobian,
        strain_energy=result.element_strain_energy,
        cauchy_stress=result.element_cauchy_stress,
        left_reaction=result.left_reaction,
        right_reaction=result.right_reaction,
    )

    global_history = np.column_stack(
        (
            result.time,
            result.mean_structural_order,
            result.mean_equilibrium_order,
            result.mean_stimulus,
            result.mean_strain_energy,
            result.mean_cauchy_stress_xx,
            result.mean_target_alignment,
            result.orientation_coherence,
            result.right_reaction[:, 0],
            result.free_dof_residual_norm,
            result.iterations,
        )
    )
    np.savetxt(
        data_directory / "equilibrium_remodeling_coupling_history.csv",
        global_history,
        delimiter=",",
        comments="",
        header=(
            "time,mean_beta,mean_beta_equilibrium,mean_stimulus,mean_strain_energy,"
            "mean_sigma_xx,mean_target_alignment,orientation_coherence,"
            "right_reaction_x,residual_inf,total_iterations"
        ),
    )

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(result.time, result.right_reaction[:, 0])
    axis.set(
        xlabel="Remodeling time",
        ylabel="Right-boundary reaction",
        title="Reaction evolution at fixed applied displacement",
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_directory / "equilibrium_remodeling_reaction.png",
        dpi=200,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(result.time, result.mean_structural_order, label="mean beta")
    axis.plot(result.time, result.mean_target_alignment, label="mean target alignment")
    axis.set(
        xlabel="Remodeling time",
        ylabel="Dimensionless value",
        title="Structural evolution under equilibrium mechanics",
        ylim=(0.0, 1.05),
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_directory / "equilibrium_remodeling_structure.png",
        dpi=200,
    )
    plt.close(figure)

    reaction_change = 100.0 * (
        result.right_reaction[-1, 0] / result.right_reaction[0, 0] - 1.0
    )
    print(f"Initial mean beta: {result.mean_structural_order[0]:.6f}")
    print(f"Final mean beta: {result.mean_structural_order[-1]:.6f}")
    print(f"Initial mean alignment: {result.mean_target_alignment[0]:.6f}")
    print(f"Final mean alignment: {result.mean_target_alignment[-1]:.6f}")
    print(f"Initial right reaction: {result.right_reaction[0, 0]:.6f}")
    print(f"Final right reaction: {result.right_reaction[-1, 0]:.6f}")
    print(f"Reaction change: {reaction_change:.3f}%")
    print(f"Maximum residual: {np.max(result.free_dof_residual_norm):.3e}")
    print(f"Minimum Jacobian: {np.min(result.element_jacobian):.6f}")


if __name__ == "__main__":
    main()
