"""Run the version-0.1 homogeneous uniaxial remodeling example."""

from __future__ import annotations

from pathlib import Path

from anisotropic_remodeling import (
    MaterialParameters,
    RemodelingParameters,
    SimulationConfig,
    run_homogeneous_remodeling,
)
from anisotropic_remodeling.visualization import plot_simulation_summary


def main() -> None:
    config = SimulationConfig(
        total_time=40.0,
        dt=0.05,
        ramp_duration=8.0,
        maximum_stretch=1.25,
        initial_fiber_angle_deg=60.0,
        initial_beta=0.1,
    )
    material = MaterialParameters(mu=10.0, kappa=1000.0, k1=2.0, k2=5.0)
    remodeling = RemodelingParameters(
        orientation_rate=0.25,
        order_rate=0.15,
        beta_min=0.1,
        beta_max=1.0,
        half_saturation=0.2,
        hill_exponent=2.0,
    )

    result = run_homogeneous_remodeling(config, material, remodeling)
    output = Path("results/figures/uniaxial_remodeling_summary.png")
    plot_simulation_summary(result, save_path=output, show=True)

    print(f"Final fiber angle: {result.fiber_angle_deg[-1]:.3f} deg")
    print(f"Final structural order beta: {result.structural_order[-1]:.3f}")
    print(f"Final axial Cauchy stress: {result.cauchy_stress[-1, 0, 0]:.3f}")
    print(f"Figure written to: {output}")


if __name__ == "__main__":
    main()
