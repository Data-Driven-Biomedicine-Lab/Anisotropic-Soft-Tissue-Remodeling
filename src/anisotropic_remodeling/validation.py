"""Synthetic validation challenges for blind calibration and held-out prediction.

Every observation is generated from a documented synthetic ground truth, split
into disjoint training and test protocols, and perturbed by reproducible
Gaussian noise.

The public challenge view exposes training observations and test inputs only.
The hidden test targets remain inside ``SyntheticValidationChallenge`` and are
used exclusively by the scoring function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .architecture import MultiFiberMaterialParameters
from .identification import MechanicalDataset, predict_dataset_stress
from .orientation import angle_to_vector, normalize_vectors, vector_to_angle

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PublicSyntheticChallenge:
    """Information made available to a calibration algorithm."""

    training_dataset: MechanicalDataset
    test_deformation_gradient: FloatArray
    test_stress_component: IntArray
    test_noise_std: FloatArray
    test_protocol_name: tuple[str, ...]
    test_load_value: FloatArray
    fiber_direction: FloatArray
    structural_order: FloatArray
    family_weights: tuple[float, ...]
    challenge_id: str

    def __post_init__(self) -> None:
        deformation = np.asarray(self.test_deformation_gradient, dtype=float)
        component = np.asarray(self.test_stress_component, dtype=np.int64)
        noise = np.asarray(self.test_noise_std, dtype=float)
        load = np.asarray(self.test_load_value, dtype=float)
        number = deformation.shape[0]
        if deformation.shape != (number, 2, 2):
            raise ValueError("test_deformation_gradient must have shape (N, 2, 2).")
        if component.shape != (number, 2):
            raise ValueError("test_stress_component must have shape (N, 2).")
        if noise.shape != (number,) or load.shape != (number,):
            raise ValueError("Test noise and load arrays must match the test inputs.")
        if len(self.test_protocol_name) != number:
            raise ValueError("One test protocol label is required per observation.")
        if np.any(np.linalg.det(deformation) <= 0.0):
            raise ValueError("All test deformation gradients must satisfy det(F) > 0.")
        if np.any(noise <= 0.0) or not np.all(np.isfinite(noise)):
            raise ValueError("test_noise_std must be finite and positive.")
        fiber = normalize_vectors(self.fiber_direction)
        beta = np.asarray(self.structural_order, dtype=float)
        if beta.shape != (fiber.shape[0],):
            raise ValueError("structural_order must match the number of fiber families.")
        if np.any((beta < 0.0) | (beta > 1.0)):
            raise ValueError("structural_order must lie in [0, 1].")
        if len(self.family_weights) != fiber.shape[0]:
            raise ValueError("family_weights must match the number of fiber families.")

        object.__setattr__(self, "test_deformation_gradient", deformation)
        object.__setattr__(self, "test_stress_component", component)
        object.__setattr__(self, "test_noise_std", noise)
        object.__setattr__(self, "test_load_value", load)
        object.__setattr__(self, "test_protocol_name", tuple(self.test_protocol_name))
        object.__setattr__(self, "fiber_direction", fiber)
        object.__setattr__(self, "structural_order", beta)

    @property
    def number_of_test_observations(self) -> int:
        return int(self.test_deformation_gradient.shape[0])

    @property
    def training_protocols(self) -> tuple[str, ...]:
        return self.training_dataset.protocols

    @property
    def test_protocols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.test_protocol_name))

    def empty_test_dataset(self) -> MechanicalDataset:
        """Return test inputs with zero placeholders instead of hidden targets."""
        return MechanicalDataset(
            deformation_gradient=self.test_deformation_gradient,
            stress_component=self.test_stress_component,
            observed_stress=np.zeros(self.number_of_test_observations),
            noise_std=self.test_noise_std,
            protocol_name=self.test_protocol_name,
            load_value=self.test_load_value,
        )


@dataclass(frozen=True, slots=True)
class SyntheticValidationChallenge:
    """Complete synthetic challenge, including hidden ground truth and targets."""

    public: PublicSyntheticChallenge
    hidden_material: MultiFiberMaterialParameters
    hidden_test_stress_clean: FloatArray
    hidden_test_stress_observed: FloatArray
    metadata: Mapping[str, str | float | int]

    def __post_init__(self) -> None:
        clean = np.asarray(self.hidden_test_stress_clean, dtype=float)
        observed = np.asarray(self.hidden_test_stress_observed, dtype=float)
        expected = (self.public.number_of_test_observations,)
        if clean.shape != expected or observed.shape != expected:
            raise ValueError("Hidden test targets must match the public test inputs.")
        if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(observed)):
            raise ValueError("Hidden test targets must be finite.")
        if self.hidden_material.number_of_families != self.public.fiber_direction.shape[0]:
            raise ValueError("Hidden material and public architecture disagree.")
        object.__setattr__(self, "hidden_test_stress_clean", clean)
        object.__setattr__(self, "hidden_test_stress_observed", observed)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Scalar scores for held-out predictions."""

    rmse: float
    mae: float
    normalized_rmse: float
    r_squared: float
    weighted_rmse: float
    fraction_within_two_sigma: float
    maximum_absolute_error: float
    number_of_observations: int


@dataclass(frozen=True, slots=True)
class ChallengeEvaluation:
    """Held-out predictions and overall plus protocol-wise scores."""

    predicted_stress: FloatArray
    overall: ValidationMetrics
    by_protocol: Mapping[str, ValidationMetrics]


def _noise_standard_deviation(
    clean_stress: FloatArray,
    protocol_name: Sequence[str],
    *,
    relative_noise: float,
    absolute_noise: float,
) -> FloatArray:
    if relative_noise < 0.0 or absolute_noise <= 0.0:
        raise ValueError("Noise scales must be non-negative with positive absolute_noise.")
    protocol = np.asarray(protocol_name, dtype=object)
    result = np.empty_like(clean_stress)
    for name in dict.fromkeys(protocol_name):
        mask = protocol == name
        scale = max(float(np.max(np.abs(clean_stress[mask]))), absolute_noise)
        result[mask] = absolute_noise + relative_noise * scale
    return result


def _make_dataset(
    deformation_gradient: FloatArray,
    stress_component: IntArray,
    protocol_name: tuple[str, ...],
    load_value: FloatArray,
    fiber_direction: FloatArray,
    structural_order: FloatArray,
    material: MultiFiberMaterialParameters,
    *,
    relative_noise: float,
    absolute_noise: float,
    random_generator: np.random.Generator,
) -> tuple[MechanicalDataset, FloatArray]:
    placeholder = MechanicalDataset(
        deformation_gradient=deformation_gradient,
        stress_component=stress_component,
        observed_stress=np.zeros(deformation_gradient.shape[0]),
        noise_std=np.ones(deformation_gradient.shape[0]),
        protocol_name=protocol_name,
        load_value=load_value,
    )
    clean = predict_dataset_stress(
        placeholder,
        fiber_direction,
        structural_order,
        material,
    )
    noise_std = _noise_standard_deviation(
        clean,
        protocol_name,
        relative_noise=relative_noise,
        absolute_noise=absolute_noise,
    )
    observed = clean + random_generator.normal(0.0, noise_std)
    return (
        MechanicalDataset(
            deformation_gradient=deformation_gradient,
            stress_component=stress_component,
            observed_stress=observed,
            noise_std=noise_std,
            protocol_name=protocol_name,
            load_value=load_value,
        ),
        clean,
    )


def _stack_protocols(
    blocks: Sequence[tuple[str, FloatArray, tuple[int, int], FloatArray]],
) -> tuple[FloatArray, IntArray, tuple[str, ...], FloatArray]:
    deformation = np.concatenate([block[1] for block in blocks], axis=0)
    component = np.concatenate(
        [
            np.broadcast_to(
                np.asarray(block[2], dtype=np.int64),
                (block[1].shape[0], 2),
            )
            for block in blocks
        ],
        axis=0,
    )
    labels = tuple(
        name
        for name, block_deformation, _, _ in blocks
        for _ in range(block_deformation.shape[0])
    )
    load = np.concatenate([block[3] for block in blocks])
    return deformation, component, labels, load


def _training_protocols() -> tuple[FloatArray, IntArray, tuple[str, ...], FloatArray]:
    axial = np.linspace(1.0, 1.16, 12)
    shear = np.linspace(0.0, 0.18, 10)
    dilation = np.linspace(1.0, 1.016, 8)

    extension_x = np.zeros((axial.size, 2, 2))
    extension_x[:, 0, 0] = axial
    extension_x[:, 1, 1] = 1.0 / axial

    extension_y = np.zeros((axial.size, 2, 2))
    extension_y[:, 0, 0] = 1.0 / axial
    extension_y[:, 1, 1] = axial

    simple_shear = np.broadcast_to(np.eye(2), (shear.size, 2, 2)).copy()
    simple_shear[:, 0, 1] = shear

    isotropic_dilation = dilation[:, None, None] * np.eye(2)
    return _stack_protocols(
        (
            ("train_uniaxial_x", extension_x, (0, 0), axial),
            ("train_uniaxial_y", extension_y, (1, 1), axial),
            ("train_simple_shear", simple_shear, (0, 1), shear),
            ("train_dilation", isotropic_dilation, (0, 0), dilation),
        )
    )


def _test_protocols() -> tuple[FloatArray, IntArray, tuple[str, ...], FloatArray]:
    off_axis_stretch = np.linspace(1.01, 1.18, 10)
    angle = np.deg2rad(32.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    off_axis = np.empty((off_axis_stretch.size, 2, 2))
    for index, stretch in enumerate(off_axis_stretch):
        principal = np.diag([stretch, 1.0 / stretch])
        off_axis[index] = rotation @ principal @ rotation.T

    combined_stretch = np.linspace(1.01, 1.14, 9)
    combined_shear = np.linspace(0.03, 0.20, 9)
    combined = np.zeros((combined_stretch.size, 2, 2))
    combined[:, 0, 0] = combined_stretch
    combined[:, 0, 1] = combined_shear
    combined[:, 1, 1] = 1.0 / combined_stretch

    compression = np.linspace(0.98, 0.86, 8)
    compression_x = np.zeros((compression.size, 2, 2))
    compression_x[:, 0, 0] = compression
    compression_x[:, 1, 1] = 1.0 / compression

    transverse_shear_value = np.linspace(0.02, 0.20, 9)
    transverse_shear = np.broadcast_to(
        np.eye(2),
        (transverse_shear_value.size, 2, 2),
    ).copy()
    transverse_shear[:, 1, 0] = transverse_shear_value

    return _stack_protocols(
        (
            ("test_off_axis_extension", off_axis, (0, 0), off_axis_stretch),
            (
                "test_combined_extension_shear",
                combined,
                (0, 1),
                combined_shear,
            ),
            ("test_axial_compression", compression_x, (0, 0), compression),
            (
                "test_transverse_shear",
                transverse_shear,
                (1, 0),
                transverse_shear_value,
            ),
        )
    )


def create_synthetic_validation_challenge(
    *,
    random_seed: int = 202609,
    relative_noise: float = 0.02,
    absolute_noise: float = 0.004,
    challenge_id: str = "ASTR-SYNTHETIC-CHALLENGE-01",
) -> SyntheticValidationChallenge:
    """Create a reproducible train/test challenge with disjoint protocols."""
    hidden_material = MultiFiberMaterialParameters(
        mu=2.65,
        kappa=210.0,
        k1=(3.45, 1.55),
        k2=(4.8, 3.15),
        family_weights=(0.68, 0.32),
    )
    fiber_direction = angle_to_vector(np.deg2rad([21.0, 108.0]))
    structural_order = np.array([0.81, 0.43], dtype=float)

    random_generator = np.random.default_rng(random_seed)
    train_inputs = _training_protocols()
    training_dataset, _ = _make_dataset(
        *train_inputs,
        fiber_direction,
        structural_order,
        hidden_material,
        relative_noise=relative_noise,
        absolute_noise=absolute_noise,
        random_generator=random_generator,
    )

    test_inputs = _test_protocols()
    test_dataset, test_clean = _make_dataset(
        *test_inputs,
        fiber_direction,
        structural_order,
        hidden_material,
        relative_noise=relative_noise,
        absolute_noise=absolute_noise,
        random_generator=random_generator,
    )

    public = PublicSyntheticChallenge(
        training_dataset=training_dataset,
        test_deformation_gradient=test_dataset.deformation_gradient,
        test_stress_component=test_dataset.stress_component,
        test_noise_std=test_dataset.noise_std,
        test_protocol_name=test_dataset.protocol_name,
        test_load_value=test_dataset.load_value,
        fiber_direction=fiber_direction,
        structural_order=structural_order,
        family_weights=hidden_material.family_weights,
        challenge_id=challenge_id,
    )
    metadata: dict[str, str | float | int] = {
        "challenge_id": challenge_id,
        "data_origin": "fully synthetic",
        "external_datasets_used": "none",
        "random_seed": int(random_seed),
        "relative_noise": float(relative_noise),
        "absolute_noise": float(absolute_noise),
        "training_protocol_count": len(public.training_protocols),
        "test_protocol_count": len(public.test_protocols),
        "license": "MIT for generator and generated benchmark files",
    }
    return SyntheticValidationChallenge(
        public=public,
        hidden_material=hidden_material,
        hidden_test_stress_clean=test_clean,
        hidden_test_stress_observed=test_dataset.observed_stress,
        metadata=metadata,
    )


def validation_metrics(
    observed: ArrayLike,
    predicted: ArrayLike,
    noise_std: ArrayLike,
) -> ValidationMetrics:
    """Compute scale-aware held-out prediction metrics."""
    observation = np.asarray(observed, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    noise = np.asarray(noise_std, dtype=float)
    if observation.shape != prediction.shape or observation.shape != noise.shape:
        raise ValueError("Observed, predicted, and noise arrays must have equal shapes.")
    if observation.ndim != 1 or observation.size < 2:
        raise ValueError("At least two scalar observations are required.")
    if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Observed and predicted values must be finite.")
    if np.any(noise <= 0.0) or not np.all(np.isfinite(noise)):
        raise ValueError("noise_std must be finite and positive.")

    error = prediction - observation
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    scale = max(
        float(np.sqrt(np.mean(observation**2))),
        float(np.ptp(observation)),
        np.finfo(float).eps,
    )
    normalized_rmse = rmse / scale
    total = float(np.sum((observation - np.mean(observation)) ** 2))
    r_squared = (
        1.0 - float(np.sum(error**2)) / total
        if total > np.finfo(float).eps
        else float("nan")
    )
    weighted_rmse = float(np.sqrt(np.mean((error / noise) ** 2)))
    within_two_sigma = float(np.mean(np.abs(error) <= 2.0 * noise))
    return ValidationMetrics(
        rmse=rmse,
        mae=mae,
        normalized_rmse=normalized_rmse,
        r_squared=r_squared,
        weighted_rmse=weighted_rmse,
        fraction_within_two_sigma=within_two_sigma,
        maximum_absolute_error=float(np.max(np.abs(error))),
        number_of_observations=int(observation.size),
    )


def evaluate_synthetic_challenge(
    challenge: SyntheticValidationChallenge,
    predicted_test_stress: ArrayLike,
    *,
    compare_to_clean_truth: bool = False,
) -> ChallengeEvaluation:
    """Score predictions without exposing hidden targets to the fitting step."""
    prediction = np.asarray(predicted_test_stress, dtype=float)
    expected = (challenge.public.number_of_test_observations,)
    if prediction.shape != expected:
        raise ValueError(f"predicted_test_stress must have shape {expected}.")
    target = (
        challenge.hidden_test_stress_clean
        if compare_to_clean_truth
        else challenge.hidden_test_stress_observed
    )
    overall = validation_metrics(
        target,
        prediction,
        challenge.public.test_noise_std,
    )
    labels = np.asarray(challenge.public.test_protocol_name, dtype=object)
    by_protocol = {}
    for protocol in challenge.public.test_protocols:
        mask = labels == protocol
        by_protocol[protocol] = validation_metrics(
            target[mask],
            prediction[mask],
            challenge.public.test_noise_std[mask],
        )
    return ChallengeEvaluation(
        predicted_stress=prediction,
        overall=overall,
        by_protocol=by_protocol,
    )


def perturb_synthetic_architecture(
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    *,
    angle_noise_std_deg: float,
    order_noise_std: float,
    random_seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Perturb a known architecture to emulate reconstruction uncertainty."""
    if angle_noise_std_deg < 0.0 or order_noise_std < 0.0:
        raise ValueError("Architecture-noise levels must be non-negative.")
    fiber = normalize_vectors(fiber_direction)
    beta = np.asarray(structural_order, dtype=float)
    if beta.shape != (fiber.shape[0],):
        raise ValueError("structural_order must match the number of families.")
    random_generator = np.random.default_rng(random_seed)
    angle = vector_to_angle(fiber)
    perturbed_angle = angle + np.deg2rad(
        random_generator.normal(0.0, angle_noise_std_deg, size=angle.shape)
    )
    perturbed_beta = np.clip(
        beta + random_generator.normal(0.0, order_noise_std, size=beta.shape),
        0.0,
        1.0,
    )
    return angle_to_vector(perturbed_angle), perturbed_beta
