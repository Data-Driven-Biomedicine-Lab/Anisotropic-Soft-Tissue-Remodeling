"""Solve the repository's polarimetry-informed finite-element example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anisotropic_remodeling import (
    FiniteElementConfig,
    MaterialParameters,
    RetardanceCalibration,
    element_centroids,
    polarimetry_to_structure,
    rectangular_quad_mesh,
    sample_nematic_image_to_elements,
    solve_displacement_controlled_equilibrium,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    measurement = np.load(root / "data" / "synthetic" / "polarimetry_maps.npz")
    structure = polarimetry_to_structure(
        measurement["azimuth_rad"],
        measurement["retardance"],
        RetardanceCalibration(0.04, 0.90, beta_min=0.05, beta_max=0.95, exponent=1.30),
        minimum_valid_retardance=0.08,
        external_valid_mask=measurement["tissue_mask"],
        coherence_window=9,
    )

    mesh = rectangular_quad_mesh(6, 3, width=2.0, height=1.2)
    centroids = element_centroids(mesh)
    image_points = np.column_stack((centroids[:, 0] - 1.0, centroids[:, 1] - 0.6))
    fiber, beta = sample_nematic_image_to_elements(
        image_points,
        measurement["x"],
        measurement["y"],
        structure.fiber_direction,
        structure.structural_order,
        structure.valid_mask,
        neighbors=12,
    )

    result = solve_displacement_controlled_equilibrium(
        mesh,
        fiber,
        beta,
        MaterialParameters(mu=5.0, kappa=100.0, k1=2.0, k2=3.0),
        FiniteElementConfig(
            axial_extension=0.06,
            load_steps=3,
            gradient_tolerance=1.0e-6,
            maximum_iterations=400,
        ),
    )

    print(f"Elements: {mesh.number_of_elements}")
    print(f"Free-DOF residual: {result.free_dof_residual_norm:.3e}")
    print(f"Minimum J: {result.element_jacobian.min():.6f}")
    print(f"Right reaction: {result.right_reaction[0]:.6f}")


if __name__ == "__main__":
    main()
