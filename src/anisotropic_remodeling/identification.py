"""Parameter identification, sensitivity, and uncertainty utilities.

The inverse problem is formulated for homogeneous multi-family material tests.
Each observation stores a deformation gradient, one observed Cauchy-stress
component, a known standard deviation, and a protocol label.

Positive material parameters are optimized in logarithmic coordinates. The
module provides local finite-difference sensitivities, covariance estimates
from the weighted least-squares Jacobian, and a deterministic parametric
bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .architecture import MultiFiberMaterialParameters, multifiber_cauchy_stress
from .orientation import normalize_vectors

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class MechanicalDataset:
    """A collection of stress observations under prescribed deformations."""

    deformation_gradient: FloatArray
    stress_component: IntArray
    observed_stress: FloatArray
    noise_std: FloatArray
    protocol_name: tuple[str, ...]
    load_value: FloatArray

    def __post_init__(self) -> None:
        deformation = np.asarray(self.deformation_gradient, dtype=float)
        component = np.asarray(self.stress_component, dtype=np.int64)
        observed = np.asarray(self.observed_stress, dtype=float)
        noise = np.asarray(self.noise_std, dtype=float)
        load = np.asarray(self.load_value, dtype=float)
        number = observed.size

        if deformation.shape != (number, 2, 2):
            raise ValueError("deformation_gradient must have shape (observations, 2, 2).")
        if component.shape != (number, 2):
            raise ValueError("stress_component must have shape (observations, 2).")
        if noise.shape != (number,) or load.shape != (number,):
            raise ValueError("noise_std and load_value must match observed_stress.")
        if len(self.protocol_name) != number:
            raise ValueError("protocol_name must contain one label per observation.")
        if not np.all(np.isfinite(deformation)) or np.any(np.linalg.det(deformation) <= 0.0):
            raise ValueError("All deformation gradients must be finite with det(F) > 0.")
        if not np.all((component >= 0) & (component <= 1)):
            raise ValueError("stress_component indices must be zero or one.")
        if not np.all(np.isfinite(observed)):
            raise ValueError("observed_stress must be finite.")
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError("noise_std must be finite and strictly positive.")
        if not np.all(np.isfinite(load)):
            raise ValueError("load_value must be finite.")

        object.__setattr__(self, "deformation_gradient", deformation)
        object.__setattr__(self, "stress_component", component)
        object.__setattr__(self, "observed_stress", observed)
        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "load_value", load)
        object.__setattr__(self, "protocol_name", tuple(str(name) for name in self.protocol_name))

    @property
    def number_of_observations(self) -> int:
        return int(self.observed_stress.size)

    @property
    def protocols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.protocol_name))

    def with_observed_stress(self, observed_stress: ArrayLike) -> "MechanicalDataset":
        observed = np.asarray(observed_stress, dtype=float)
        if observed.shape != self.observed_stress.shape:
            raise ValueError("Replacement observations must preserve the dataset shape.")
        return replace(self, observed_stress=observed)

    def subset(self, protocols: Sequence[str]) -> "MechanicalDataset":
        selected = {str(name) for name in protocols}
        mask = np.asarray([name in selected for name in self.protocol_name], dtype=bool)
        if not np.any(mask):
            raise ValueError("No observations match the requested protocols.")
        return MechanicalDataset(
            deformation_gradient=self.deformation_gradient[mask],
            stress_component=self.stress_component[mask],
            observed_stress=self.observed_stress[mask],
            noise_std=self.noise_std[mask],
            protocol_name=tuple(np.asarray(self.protocol_name, dtype=object)[mask]),
            load_value=self.load_value[mask],
        )


@dataclass(frozen=True, slots=True)
class MaterialParameterMap:
    """Map between a positive parameter vector and a multi-family material."""

    number_of_families: int
    family_weights: tuple[float, ...]
    fixed_kappa: float = 1000.0
    identify_kappa: bool = False

    def __post_init__(self) -> None:
        if self.number_of_families < 1:
            raise ValueError("number_of_families must be positive.")
        if len(self.family_weights) != self.number_of_families:
            raise ValueError("family_weights must match number_of_families.")
        weight = np.asarray(self.family_weights, dtype=float)
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0) or np.sum(weight) <= 0.0:
            raise ValueError("family_weights must be finite, non-negative, and not all zero.")
        if not np.isfinite(self.fixed_kappa) or self.fixed_kappa <= 0.0:
            raise ValueError("fixed_kappa must be finite and strictly positive.")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ["mu"]
        names.extend(f"k1_{family + 1}" for family in range(self.number_of_families))
        names.extend(f"k2_{family + 1}" for family in range(self.number_of_families))
        if self.identify_kappa:
            names.append("kappa")
        return tuple(names)

    @property
    def size(self) -> int:
        return len(self.parameter_names)

    def pack(self, material: MultiFiberMaterialParameters) -> FloatArray:
        if material.number_of_families != self.number_of_families:
            raise ValueError("Material family count does not match the parameter map.")
        values: list[float] = [material.mu]
        values.extend(material.k1)
        values.extend(material.k2)
        if self.identify_kappa:
            values.append(material.kappa)
        vector = np.asarray(values, dtype=float)
        if np.any(vector <= 0.0) or not np.all(np.isfinite(vector)):
            raise ValueError("All identifiable material parameters must be positive and finite.")
        return vector

    def unpack(self, values: ArrayLike) -> MultiFiberMaterialParameters:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(f"Parameter vector must have shape ({self.size},).")
        if np.any(vector <= 0.0) or not np.all(np.isfinite(vector)):
            raise ValueError("All identifiable parameters must be positive and finite.")
        family_count = self.number_of_families
        mu = float(vector[0])
        k1 = tuple(float(value) for value in vector[1 : 1 + family_count])
        k2 = tuple(
            float(value)
            for value in vector[1 + family_count : 1 + 2 * family_count]
        )
        kappa = (
            float(vector[-1])
            if self.identify_kappa
            else float(self.fixed_kappa)
        )
        return MultiFiberMaterialParameters(
            mu=mu,
            kappa=kappa,
            k1=k1,
            k2=k2,
            family_weights=self.family_weights,
        )


@dataclass(frozen=True, slots=True)
class MaterialFitResult:
    """Weighted nonlinear least-squares calibration result."""

    material: MultiFiberMaterialParameters
    parameter_names: tuple[str, ...]
    parameter_vector: FloatArray
    standard_error: FloatArray
    covariance: FloatArray
    correlation: FloatArray
    predicted_stress: FloatArray
    weighted_residual: FloatArray
    objective: float
    degrees_of_freedom: int
    jacobian_log_parameters: FloatArray
    singular_values: FloatArray
    condition_number: float
    success: bool
    message: str
    number_of_function_evaluations: int


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Parametric-bootstrap material estimates."""

    parameter_names: tuple[str, ...]
    parameter_samples: FloatArray
    successful: NDArray[np.bool_]

    @property
    def successful_samples(self) -> FloatArray:
        return self.parameter_samples[self.successful]

    def percentile_interval(
        self,
        lower: float = 2.5,
        upper: float = 97.5,
    ) -> tuple[FloatArray, FloatArray]:
        if not (0.0 <= lower < upper <= 100.0):
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
        samples = self.successful_samples
        if samples.size == 0:
            raise RuntimeError("No successful bootstrap samples are available.")
        return (
            np.percentile(samples, lower, axis=0),
            np.percentile(samples, upper, axis=0),
        )


def predict_dataset_stress(
    dataset: MechanicalDataset,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    material: MultiFiberMaterialParameters,
) -> FloatArray:
    """Predict the requested Cauchy-stress component for every observation."""
    fiber = normalize_vectors(fiber_direction)
    beta = np.asarray(structural_order, dtype=float)
    if fiber.shape != (material.number_of_families, 2):
        raise ValueError("fiber_direction must have shape (families, 2).")
    if beta.shape != (material.number_of_families,):
        raise ValueError("structural_order must have shape (families,).")
    if not np.all(np.isfinite(beta)) or np.any((beta < 0.0) | (beta > 1.0)):
        raise ValueError("structural_order must be finite and lie in [0, 1].")

    stress = multifiber_cauchy_stress(
        dataset.deformation_gradient,
        fiber,
        beta,
        material,
    )
    observation_index = np.arange(dataset.number_of_observations)
    return stress[
        observation_index,
        dataset.stress_component[:, 0],
        dataset.stress_component[:, 1],
    ]


def build_multiaxial_protocol_dataset(
    material: MultiFiberMaterialParameters,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    *,
    axial_stretches: ArrayLike = np.linspace(1.0, 1.18, 13),
    shear_values: ArrayLike = np.linspace(0.0, 0.20, 11),
    dilation_values: ArrayLike | None = None,
    relative_noise: float = 0.015,
    absolute_noise: float = 0.003,
    random_seed: int = 1234,
) -> MechanicalDataset:
    """Generate a reproducible synthetic multi-protocol mechanical dataset.

    Protocols
    ---------
    ``uniaxial_x``
        ``F = diag(lambda, 1/lambda)``, observe ``sigma_xx``.
    ``uniaxial_y``
        ``F = diag(1/lambda, lambda)``, observe ``sigma_yy``.
    ``simple_shear``
        ``F = [[1, gamma], [0, 1]]``, observe ``sigma_xy``.
    ``dilation`` (optional)
        ``F = s*I``, observe ``sigma_xx``. This protocol introduces sensitivity
        to the volumetric parameter ``kappa``.
    """
    if relative_noise < 0.0 or absolute_noise <= 0.0:
        raise ValueError("relative_noise must be non-negative and absolute_noise positive.")

    blocks: list[tuple[str, FloatArray, tuple[int, int], FloatArray]] = []
    axial = np.asarray(axial_stretches, dtype=float)
    shear = np.asarray(shear_values, dtype=float)
    if axial.ndim != 1 or shear.ndim != 1:
        raise ValueError("Protocol load arrays must be one-dimensional.")
    if np.any(axial <= 0.0):
        raise ValueError("Axial stretches must be strictly positive.")

    deformation_x = np.zeros((axial.size, 2, 2), dtype=float)
    deformation_x[:, 0, 0] = axial
    deformation_x[:, 1, 1] = 1.0 / axial
    blocks.append(("uniaxial_x", deformation_x, (0, 0), axial))

    deformation_y = np.zeros((axial.size, 2, 2), dtype=float)
    deformation_y[:, 0, 0] = 1.0 / axial
    deformation_y[:, 1, 1] = axial
    blocks.append(("uniaxial_y", deformation_y, (1, 1), axial))

    deformation_shear = np.broadcast_to(np.eye(2), (shear.size, 2, 2)).copy()
    deformation_shear[:, 0, 1] = shear
    blocks.append(("simple_shear", deformation_shear, (0, 1), shear))

    if dilation_values is not None:
        dilation = np.asarray(dilation_values, dtype=float)
        if dilation.ndim != 1 or np.any(dilation <= 0.0):
            raise ValueError("dilation_values must be a positive one-dimensional array.")
        deformation_dilation = dilation[:, None, None] * np.eye(2)
        blocks.append(("dilation", deformation_dilation, (0, 0), dilation))

    deformation = np.concatenate([block[1] for block in blocks], axis=0)
    component = np.concatenate(
        [
            np.broadcast_to(np.asarray(block[2], dtype=np.int64), (block[1].shape[0], 2))
            for block in blocks
        ],
        axis=0,
    )
    protocol_name = tuple(
        name
        for name, block_deformation, _, _ in blocks
        for _ in range(block_deformation.shape[0])
    )
    load_value = np.concatenate([block[3] for block in blocks], axis=0)

    clean_dataset = MechanicalDataset(
        deformation_gradient=deformation,
        stress_component=component,
        observed_stress=np.zeros(deformation.shape[0], dtype=float),
        noise_std=np.ones(deformation.shape[0], dtype=float),
        protocol_name=protocol_name,
        load_value=load_value,
    )
    clean_stress = predict_dataset_stress(
        clean_dataset,
        fiber_direction,
        structural_order,
        material,
    )

    noise_std = np.empty_like(clean_stress)
    offset = 0
    for _, block_deformation, _, _ in blocks:
        count = block_deformation.shape[0]
        block_stress = clean_stress[offset : offset + count]
        protocol_scale = max(float(np.max(np.abs(block_stress))), absolute_noise)
        noise_std[offset : offset + count] = (
            absolute_noise + relative_noise * protocol_scale
        )
        offset += count

    rng = np.random.default_rng(random_seed)
    observed = clean_stress + rng.normal(0.0, noise_std)
    return MechanicalDataset(
        deformation_gradient=deformation,
        stress_component=component,
        observed_stress=observed,
        noise_std=noise_std,
        protocol_name=protocol_name,
        load_value=load_value,
    )


def weighted_residuals(
    parameter_values: ArrayLike,
    dataset: MechanicalDataset,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameter_map: MaterialParameterMap,
) -> FloatArray:
    """Return residuals normalized by known observation standard deviations."""
    material = parameter_map.unpack(parameter_values)
    prediction = predict_dataset_stress(
        dataset,
        fiber_direction,
        structural_order,
        material,
    )
    return (prediction - dataset.observed_stress) / dataset.noise_std


def local_sensitivity_matrix(
    dataset: MechanicalDataset,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    material: MultiFiberMaterialParameters,
    parameter_map: MaterialParameterMap,
    *,
    relative_step: float = 1.0e-5,
    normalized: bool = False,
) -> FloatArray:
    """Compute central finite-difference sensitivities to physical parameters."""
    if relative_step <= 0.0 or not np.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive.")
    base = parameter_map.pack(material)
    sensitivity = np.empty((dataset.number_of_observations, base.size), dtype=float)

    for parameter_index, value in enumerate(base):
        step = relative_step * max(abs(float(value)), 1.0)
        plus = base.copy()
        minus = base.copy()
        plus[parameter_index] += step
        minus[parameter_index] = max(minus[parameter_index] - step, 0.5 * value)
        prediction_plus = predict_dataset_stress(
            dataset,
            fiber_direction,
            structural_order,
            parameter_map.unpack(plus),
        )
        prediction_minus = predict_dataset_stress(
            dataset,
            fiber_direction,
            structural_order,
            parameter_map.unpack(minus),
        )
        sensitivity[:, parameter_index] = (
            prediction_plus - prediction_minus
        ) / (plus[parameter_index] - minus[parameter_index])

    if normalized:
        prediction = predict_dataset_stress(
            dataset,
            fiber_direction,
            structural_order,
            material,
        )
        observation_scale = np.maximum(np.abs(prediction), dataset.noise_std)
        sensitivity = sensitivity * base[None, :] / observation_scale[:, None]
    return sensitivity


def fit_material_parameters(
    dataset: MechanicalDataset,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameter_map: MaterialParameterMap,
    *,
    initial_values: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    maximum_function_evaluations: int = 3000,
) -> MaterialFitResult:
    """Fit positive material parameters by bounded weighted least squares."""
    initial = np.asarray(initial_values, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    expected_shape = (parameter_map.size,)
    if initial.shape != expected_shape or lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError(f"Initial values and bounds must have shape {expected_shape}.")
    if np.any(lower <= 0.0) or np.any(upper <= lower):
        raise ValueError("Bounds must be positive and satisfy lower < upper.")
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError("Initial values must lie strictly inside the bounds.")

    def residual_log(log_values: FloatArray) -> FloatArray:
        values = np.exp(log_values)
        return weighted_residuals(
            values,
            dataset,
            fiber_direction,
            structural_order,
            parameter_map,
        )

    optimization = least_squares(
        residual_log,
        np.log(initial),
        bounds=(np.log(lower), np.log(upper)),
        method="trf",
        x_scale="jac",
        jac="3-point",
        max_nfev=maximum_function_evaluations,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-11,
    )
    parameter_vector = np.exp(optimization.x)
    material = parameter_map.unpack(parameter_vector)
    prediction = predict_dataset_stress(
        dataset,
        fiber_direction,
        structural_order,
        material,
    )
    weighted = (prediction - dataset.observed_stress) / dataset.noise_std
    degrees_of_freedom = max(dataset.number_of_observations - parameter_map.size, 1)
    objective = float(np.dot(weighted, weighted))
    residual_variance = objective / degrees_of_freedom

    jacobian_log = np.asarray(optimization.jac, dtype=float)
    information = jacobian_log.T @ jacobian_log
    covariance_log = residual_variance * np.linalg.pinv(information, rcond=1.0e-12)
    transformation = np.diag(parameter_vector)
    covariance = transformation @ covariance_log @ transformation
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_error, standard_error)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 1.0)

    singular_values = np.linalg.svd(jacobian_log, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > np.finfo(float).eps * singular_values[0]
        else float("inf")
    )

    return MaterialFitResult(
        material=material,
        parameter_names=parameter_map.parameter_names,
        parameter_vector=parameter_vector,
        standard_error=standard_error,
        covariance=covariance,
        correlation=correlation,
        predicted_stress=prediction,
        weighted_residual=weighted,
        objective=objective,
        degrees_of_freedom=degrees_of_freedom,
        jacobian_log_parameters=jacobian_log,
        singular_values=singular_values,
        condition_number=condition_number,
        success=bool(optimization.success),
        message=str(optimization.message),
        number_of_function_evaluations=int(optimization.nfev),
    )


def parametric_bootstrap_material_fit(
    dataset: MechanicalDataset,
    fiber_direction: ArrayLike,
    structural_order: ArrayLike,
    parameter_map: MaterialParameterMap,
    reference_fit: MaterialFitResult,
    *,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    number_of_samples: int = 100,
    random_seed: int = 2026,
    maximum_function_evaluations: int = 1500,
) -> BootstrapResult:
    """Generate noisy replicas around the fitted model and refit each one."""
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive.")
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.shape != (parameter_map.size,) or upper.shape != (parameter_map.size,):
        raise ValueError("Bounds do not match the parameter map.")

    rng = np.random.default_rng(random_seed)
    samples = np.full((number_of_samples, parameter_map.size), np.nan, dtype=float)
    successful = np.zeros(number_of_samples, dtype=bool)
    reference_prediction = reference_fit.predicted_stress

    for sample_index in range(number_of_samples):
        synthetic_observation = reference_prediction + rng.normal(
            0.0,
            dataset.noise_std,
        )
        bootstrap_dataset = dataset.with_observed_stress(synthetic_observation)
        initial = np.clip(
            reference_fit.parameter_vector
            * np.exp(rng.normal(0.0, 0.03, parameter_map.size)),
            lower * 1.001,
            upper * 0.999,
        )
        try:
            fit = fit_material_parameters(
                bootstrap_dataset,
                fiber_direction,
                structural_order,
                parameter_map,
                initial_values=initial,
                lower_bounds=lower,
                upper_bounds=upper,
                maximum_function_evaluations=maximum_function_evaluations,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        samples[sample_index] = fit.parameter_vector
        successful[sample_index] = fit.success and np.all(np.isfinite(fit.parameter_vector))

    return BootstrapResult(
        parameter_names=parameter_map.parameter_names,
        parameter_samples=samples,
        successful=successful,
    )
