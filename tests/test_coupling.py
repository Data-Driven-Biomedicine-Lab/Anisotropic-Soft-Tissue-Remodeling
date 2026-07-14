import numpy as np
import pytest

from anisotropic_remodeling import (
    MaterialParameters,
    PolarimetryRemodelingConfig,
    RemodelingParameters,
    angle_to_vector,
    run_polarimetry_initialized_remodeling,
)


def small_fields():
    x_coordinates = np.linspace(-1.0, 1.0, 9)
    y_coordinates = np.linspace(-0.75, 0.75, 7)
    x, y = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    mask = (x / 1.0) ** 2 + (y / 0.75) ** 2 <= 0.9**2
    angle = np.deg2rad(35.0 + 15.0 * x)
    fiber = angle_to_vector(angle)
    beta = np.clip(0.45 - 0.15 * np.exp(-4.0 * (x**2 + y**2)), 0.0, 1.0)
    fiber[~mask] = np.nan
    beta[~mask] = np.nan
    return x, y, fiber, beta, mask


def test_invalid_pixels_remain_nan_and_valid_pixels_are_finite():
    x, y, fiber, beta, mask = small_fields()
    result = run_polarimetry_initialized_remodeling(
        x,
        y,
        fiber,
        beta,
        mask,
        config=PolarimetryRemodelingConfig(
            total_time=0.4,
            dt=0.1,
            ramp_duration=0.2,
            maximum_stretch=1.10,
            maximum_shear=0.2,
            half_height=0.75,
        ),
        snapshot_times=(0.0, 0.4),
    )

    assert np.all(np.isnan(result.structural_order[:, ~mask]))
    assert np.all(np.isfinite(result.structural_order[:, mask]))
    assert np.all(np.isnan(result.fiber_direction[:, ~mask]))
    assert np.all(np.isfinite(result.cauchy_stress[:, mask]))


def test_zero_rates_preserve_initial_structure():
    x, y, fiber, beta, mask = small_fields()
    result = run_polarimetry_initialized_remodeling(
        x,
        y,
        fiber,
        beta,
        mask,
        config=PolarimetryRemodelingConfig(
            total_time=0.4,
            dt=0.1,
            ramp_duration=0.2,
            maximum_stretch=1.10,
            maximum_shear=0.2,
            half_height=0.75,
        ),
        remodeling=RemodelingParameters(orientation_rate=0.0, order_rate=0.0),
        snapshot_times=(0.0, 0.4),
    )

    np.testing.assert_allclose(
        result.fiber_direction[0, mask],
        result.fiber_direction[-1, mask],
    )
    np.testing.assert_allclose(
        result.structural_order[0, mask],
        result.structural_order[-1, mask],
    )


def test_physical_bounds_and_symmetric_cauchy_stress():
    x, y, fiber, beta, mask = small_fields()
    result = run_polarimetry_initialized_remodeling(
        x,
        y,
        fiber,
        beta,
        mask,
        config=PolarimetryRemodelingConfig(
            total_time=0.4,
            dt=0.1,
            ramp_duration=0.2,
            maximum_stretch=1.10,
            maximum_shear=0.2,
            half_height=0.75,
        ),
        material=MaterialParameters(),
        snapshot_times=(0.0, 0.4),
    )

    valid_beta = result.structural_order[:, mask]
    assert np.all((valid_beta >= 0.0) & (valid_beta <= 1.0))
    norms = np.linalg.norm(result.fiber_direction[:, mask], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        result.cauchy_stress[:, mask],
        np.swapaxes(result.cauchy_stress[:, mask], -1, -2),
        atol=1.0e-12,
    )
    determinants = np.linalg.det(result.deformation_gradient[:, mask])
    assert np.all(determinants > 0.0)


def test_rejects_nonfinite_values_on_valid_pixels():
    x, y, fiber, beta, mask = small_fields()
    location = tuple(np.argwhere(mask)[0])
    beta[location] = np.nan

    with pytest.raises(ValueError, match="Valid structural-order"):
        run_polarimetry_initialized_remodeling(x, y, fiber, beta, mask)


def test_rejects_empty_mask():
    x, y, fiber, beta, mask = small_fields()
    with pytest.raises(ValueError, match="at least one valid pixel"):
        run_polarimetry_initialized_remodeling(x, y, fiber, beta, np.zeros_like(mask))


def test_nondivisible_time_step_reaches_exact_final_time():
    x, y, fiber, beta, mask = small_fields()
    result = run_polarimetry_initialized_remodeling(
        x,
        y,
        fiber,
        beta,
        mask,
        config=PolarimetryRemodelingConfig(
            total_time=0.45,
            dt=0.2,
            ramp_duration=0.25,
            maximum_stretch=1.05,
            maximum_shear=0.1,
            half_height=0.75,
        ),
    )

    assert result.time[-1] == pytest.approx(0.45)
    assert np.all(np.diff(result.time) > 0.0)
    assert np.max(np.diff(result.time)) <= 0.2 + 1.0e-15
    np.testing.assert_allclose(result.snapshot_time[-1], 0.45)
