"""Run the version-0.7 multi-family and regularization benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anisotropic_remodeling import (
    FiniteElementConfig,
    MultiFiberMaterialParameters,
    angle_to_vector,
    element_centroids,
    nematic_graph_roughness,
    rectangular_quad_mesh,
    regularize_nematic_field,
    regularize_scalar_field,
    scalar_graph_roughness,
    solve_multifiber_displacement_controlled_equilibrium,
    vector_to_angle,
)


def main() -> None:
    mesh = rectangular_quad_mesh(5, 3, width=2.0, height=1.0)
    centroid = element_centroids(mesh)
    x = centroid[:, 0] / mesh.width
    y = centroid[:, 1] / mesh.height
    rng = np.random.default_rng(17)

    primary_angle = np.deg2rad(
        25.0
        + 18.0 * np.sin(np.pi * x) * np.cos(np.pi * (y - 0.5))
        + rng.normal(0.0, 7.0, mesh.number_of_elements)
    )
    secondary_angle = primary_angle + np.deg2rad(72.0)
    directions = np.stack(
        (angle_to_vector(primary_angle), angle_to_vector(secondary_angle)),
        axis=1,
    )

    defect = np.exp(-0.5 * (((x - 0.62) / 0.18) ** 2 + ((y - 0.55) / 0.22) ** 2))
    beta = np.column_stack(
        (
            np.clip(0.72 - 0.30 * defect + rng.normal(0.0, 0.05, x.size), 0.05, 1.0),
            np.clip(0.38 - 0.12 * defect + rng.normal(0.0, 0.04, x.size), 0.05, 1.0),
        )
    )

    regularized_directions = np.stack(
        [
            regularize_nematic_field(mesh, directions[:, family], strength=0.8)
            for family in range(2)
        ],
        axis=1,
    )
    regularized_beta = regularize_scalar_field(
        mesh,
        beta,
        strength=0.8,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    material = MultiFiberMaterialParameters(
        mu=2.0,
        kappa=80.0,
        k1=(2.2, 1.4),
        k2=(4.0, 3.5),
        family_weights=(0.65, 0.35),
    )
    result = solve_multifiber_displacement_controlled_equilibrium(
        mesh,
        regularized_directions,
        regularized_beta,
        material,
        FiniteElementConfig(
            axial_extension=0.05,
            load_steps=3,
            gradient_tolerance=3e-7,
            maximum_iterations=450,
        ),
    )

    print(f"Right reaction: {result.right_reaction[0]:.6f}")
    print(f"Residual: {result.free_dof_residual_norm:.3e}")
    print(f"Minimum J: {np.min(result.element_jacobian):.6f}")
    print(
        "Primary orientation roughness: "
        f"{nematic_graph_roughness(mesh, directions[:, 0]):.6f} -> "
        f"{nematic_graph_roughness(mesh, regularized_directions[:, 0]):.6f}"
    )
    print(
        "Primary beta roughness: "
        f"{scalar_graph_roughness(mesh, beta[:, 0]):.6f} -> "
        f"{scalar_graph_roughness(mesh, regularized_beta[:, 0]):.6f}"
    )

    output = Path("results/figures")
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    angle_deg = np.rad2deg(vector_to_angle(regularized_directions[:, 0]))
    scatter = axis.scatter(
        centroid[:, 0],
        centroid[:, 1],
        c=angle_deg,
        s=180,
    )
    axis.set(
        xlabel="X",
        ylabel="Y",
        title="Regularized primary-family angle [deg]",
        aspect="equal",
    )
    figure.colorbar(scatter, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "multifiber_regularized_primary_angle.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
