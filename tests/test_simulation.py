import numpy as np

from anisotropic_remodeling.simulation import SimulationConfig, run_homogeneous_remodeling


def test_reference_simulation_runs_and_remodels() -> None:
    result = run_homogeneous_remodeling(
        SimulationConfig(total_time=2.0, dt=0.02, ramp_duration=0.5)
    )
    assert result.time.ndim == 1
    assert result.deformation_gradient.shape == (result.time.size, 2, 2)
    assert result.cauchy_stress.shape == (result.time.size, 2, 2)
    assert np.all(np.isfinite(result.cauchy_stress))
    assert result.fiber_angle_deg[-1] < result.fiber_angle_deg[0]
    assert result.structural_order[-1] > result.structural_order[0]
