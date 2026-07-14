import numpy as np

from anisotropic_remodeling.orientation import angle_to_vector, nematic_angle_difference, vector_to_angle
from anisotropic_remodeling.remodeling import update_fiber_orientation, update_structural_order


def test_orientation_update_reduces_nematic_distance() -> None:
    current = angle_to_vector(np.deg2rad(70.0))
    target = angle_to_vector(np.deg2rad(5.0))
    updated = update_fiber_orientation(current, target, rate=0.8, dt=0.5)

    old_distance = abs(nematic_angle_difference(vector_to_angle(current), vector_to_angle(target)))
    new_distance = abs(nematic_angle_difference(vector_to_angle(updated), vector_to_angle(target)))
    assert new_distance < old_distance


def test_orientation_update_is_sign_invariant() -> None:
    current = np.array([0.0, 1.0])
    target = np.array([1.0, 0.0])
    positive = update_fiber_orientation(current, target, rate=1.0, dt=0.2)
    negative = update_fiber_orientation(-current, -target, rate=1.0, dt=0.2)
    np.testing.assert_allclose(np.outer(positive, positive), np.outer(negative, negative), atol=1e-12)


def test_order_update_matches_exact_solution() -> None:
    updated = update_structural_order(0.2, 0.8, rate=0.5, dt=2.0)
    expected = 0.8 + (0.2 - 0.8) * np.exp(-1.0)
    np.testing.assert_allclose(updated, expected, atol=1e-12)


def test_orientation_update_accepts_spatial_rate_field() -> None:
    fiber = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    target = np.array([[[0.0, 1.0], [0.0, 1.0]]])
    rate = np.array([[0.0, 1.0]])
    updated = update_fiber_orientation(fiber, target, rate=rate, dt=1.0)
    assert np.allclose(updated[0, 0], fiber[0, 0])
    assert not np.allclose(updated[0, 1], fiber[0, 1])
