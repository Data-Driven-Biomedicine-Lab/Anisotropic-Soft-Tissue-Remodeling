import numpy as np

from anisotropic_remodeling.orientation import (
    angle_to_vector,
    nematic_angle_difference,
    orientation_tensor,
    vector_to_angle,
)


def test_angle_vector_round_trip_is_nematic() -> None:
    angles = np.array([-0.2, 0.0, 0.7, np.pi, 1.8 * np.pi])
    recovered = vector_to_angle(angle_to_vector(angles))
    difference = nematic_angle_difference(angles, recovered)
    np.testing.assert_allclose(difference, 0.0, atol=1e-12)


def test_orientation_tensor_is_sign_invariant() -> None:
    vector = np.array([0.6, 0.8])
    np.testing.assert_allclose(orientation_tensor(vector), orientation_tensor(-vector))


def test_orientation_tensor_has_unit_trace() -> None:
    tensor = orientation_tensor(np.array([2.0, 1.0]))
    np.testing.assert_allclose(np.trace(tensor), 1.0, atol=1e-12)
