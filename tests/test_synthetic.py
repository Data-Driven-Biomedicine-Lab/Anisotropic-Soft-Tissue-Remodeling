import numpy as np

from anisotropic_remodeling.synthetic import synthetic_polarimetry_benchmark


def test_synthetic_polarimetry_benchmark_is_reproducible() -> None:
    first = synthetic_polarimetry_benchmark(random_seed=11)
    second = synthetic_polarimetry_benchmark(random_seed=11)

    np.testing.assert_allclose(first.observed_azimuth_rad, second.observed_azimuth_rad)
    np.testing.assert_allclose(first.observed_retardance, second.observed_retardance)
    assert first.x.shape == (41, 81)
    assert np.min(first.true_structural_order) >= 0.0
    assert np.max(first.true_structural_order) <= 1.0
    assert np.any(first.observed_retardance < first.calibration.lower_retardance)
