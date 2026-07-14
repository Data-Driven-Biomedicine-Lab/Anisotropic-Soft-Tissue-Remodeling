import numpy as np

from anisotropic_remodeling.material import MaterialParameters
from anisotropic_remodeling.orientation import normalize_vectors
from anisotropic_remodeling.remodeling import RemodelingParameters
from anisotropic_remodeling.spatial import (
    SpatialSimulationConfig,
    compatible_shear_extension_deformation,
    rectangular_grid,
    run_spatial_remodeling,
    synthetic_fiber_field,
    synthetic_structural_order_field,
)


def test_compatible_deformation_is_area_preserving() -> None:
    config = SpatialSimulationConfig(nx=13, ny=9)
    _, y = rectangular_grid(config)
    deformation = compatible_shear_extension_deformation(
        y,
        config.ramp_duration,
        half_height=config.half_height,
        maximum_stretch=config.maximum_stretch,
        maximum_shear=config.maximum_shear,
        ramp_duration=config.ramp_duration,
    )
    assert deformation.shape == (config.ny, config.nx, 2, 2)
    assert np.allclose(np.linalg.det(deformation), 1.0, atol=1.0e-12)


def test_synthetic_fields_are_physical() -> None:
    config = SpatialSimulationConfig(nx=13, ny=9)
    x, y = rectangular_grid(config)
    fiber = synthetic_fiber_field(
        x,
        y,
        half_width=config.half_width,
        half_height=config.half_height,
        mean_angle_deg=config.mean_fiber_angle_deg,
        angle_amplitude_deg=config.angle_amplitude_deg,
    )
    beta = synthetic_structural_order_field(
        x,
        y,
        background=config.beta_background,
        defect_depth=config.beta_defect_depth,
        defect_center_x=config.defect_center_x,
        defect_center_y=config.defect_center_y,
        defect_width_x=config.defect_width_x,
        defect_width_y=config.defect_width_y,
    )
    assert np.allclose(np.linalg.norm(normalize_vectors(fiber), axis=-1), 1.0)
    assert np.all((beta >= 0.0) & (beta <= 1.0))
    assert np.min(beta) < np.max(beta)


def test_spatial_simulation_is_bounded_and_symmetric() -> None:
    config = SpatialSimulationConfig(
        nx=15,
        ny=11,
        total_time=1.0,
        dt=0.1,
        ramp_duration=0.5,
    )
    result = run_spatial_remodeling(
        config,
        MaterialParameters(),
        RemodelingParameters(),
        snapshot_times=(0.0, 0.5, 1.0),
    )
    assert result.fiber_direction.shape == (3, config.ny, config.nx, 2)
    assert result.structural_order.shape == (3, config.ny, config.nx)
    assert np.all((result.structural_order >= 0.0) & (result.structural_order <= 1.0))
    assert np.allclose(np.linalg.norm(result.fiber_direction, axis=-1), 1.0)
    assert np.allclose(
        result.cauchy_stress,
        np.swapaxes(result.cauchy_stress, -1, -2),
        atol=1.0e-10,
    )
    assert np.all(np.linalg.det(result.deformation_gradient) > 0.0)
    assert result.mean_structural_order[-1] > result.mean_structural_order[0]
    assert result.mean_target_alignment[-1] > result.mean_target_alignment[1]
