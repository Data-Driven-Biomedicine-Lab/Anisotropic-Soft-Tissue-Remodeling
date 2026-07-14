import numpy as np
import pytest

from anisotropic_remodeling.polarimetry import (
    RetardanceCalibration,
    canonicalize_nematic_azimuth,
    local_nematic_coherence,
    polarimetry_to_structure,
    retardance_to_order_proxy,
)


def test_calibration_maps_endpoints_and_clips() -> None:
    calibration = RetardanceCalibration(
        lower_retardance=0.2,
        upper_retardance=0.8,
        beta_min=0.1,
        beta_max=0.9,
        exponent=1.0,
    )
    signal = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    beta = retardance_to_order_proxy(signal, calibration)
    assert np.allclose(beta, [0.1, 0.1, 0.5, 0.9, 0.9])


def test_invalid_calibration_is_rejected() -> None:
    with pytest.raises(ValueError):
        RetardanceCalibration(lower_retardance=1.0, upper_retardance=1.0)
    with pytest.raises(ValueError):
        RetardanceCalibration(lower_retardance=0.0, upper_retardance=1.0, exponent=0.0)


def test_canonical_azimuth_is_pi_periodic() -> None:
    angle = np.deg2rad(np.array([[5.0, 175.0], [35.0, 90.0]]))
    assert np.allclose(
        canonicalize_nematic_azimuth(angle),
        canonicalize_nematic_azimuth(angle + np.pi),
    )


def test_local_coherence_distinguishes_aligned_and_orthogonal_fields() -> None:
    aligned = np.full((5, 5), np.deg2rad(25.0))
    aligned_coherence = local_nematic_coherence(aligned, window_size=5)
    assert np.isclose(aligned_coherence[2, 2], 1.0)

    orthogonal = np.zeros((5, 5), dtype=float)
    orthogonal[:, :2] = 0.0
    orthogonal[:, 2:] = np.pi / 2.0
    weights = np.ones_like(orthogonal)
    weights[:, 2] = 0.0
    orthogonal_coherence = local_nematic_coherence(
        orthogonal,
        window_size=5,
        weights=weights,
    )
    assert np.isclose(orthogonal_coherence[2, 2], 0.0, atol=1.0e-12)


def test_polarimetry_conversion_produces_projectors_and_respects_mask() -> None:
    azimuth = np.deg2rad(
        np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0],
            ]
        )
    )
    retardance = np.full((3, 3), 0.6)
    retardance[0, 0] = 0.01
    calibration = RetardanceCalibration(0.1, 1.0, beta_min=0.05, beta_max=0.95)

    result = polarimetry_to_structure(
        azimuth,
        retardance,
        calibration,
        minimum_valid_retardance=0.1,
        coherence_window=3,
    )

    assert not result.valid_mask[0, 0]
    assert np.isnan(result.structural_order[0, 0])
    assert np.all((result.structural_order[result.valid_mask] >= 0.0))
    assert np.all((result.structural_order[result.valid_mask] <= 1.0))

    tensor = result.structure_tensor[result.valid_mask]
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    idempotence_error = np.linalg.norm(tensor @ tensor - tensor, axis=(-2, -1))
    assert np.allclose(trace, 1.0)
    assert np.allclose(idempotence_error, 0.0, atol=1.0e-12)


def test_structure_is_invariant_to_random_pi_flips() -> None:
    rng = np.random.default_rng(17)
    azimuth = rng.uniform(0.0, np.pi, size=(11, 9))
    retardance = rng.uniform(0.2, 0.9, size=(11, 9))
    flips = rng.integers(0, 2, size=(11, 9)) * np.pi
    calibration = RetardanceCalibration(0.1, 1.0)

    original = polarimetry_to_structure(
        azimuth,
        retardance,
        calibration,
        coherence_window=3,
    )
    flipped = polarimetry_to_structure(
        azimuth + flips,
        retardance,
        calibration,
        coherence_window=3,
    )

    assert np.allclose(original.structure_tensor, flipped.structure_tensor)
    assert np.allclose(original.local_coherence, flipped.local_coherence)
    assert np.allclose(original.structural_order, flipped.structural_order)
