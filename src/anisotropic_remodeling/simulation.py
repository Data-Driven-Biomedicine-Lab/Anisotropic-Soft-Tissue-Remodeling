"""Homogeneous reference simulation for version 0.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .loading import area_preserving_uniaxial_deformation
from .material import MaterialParameters, cauchy_stress, strain_energy_density
from .orientation import angle_to_vector, vector_to_angle
from .remodeling import RemodelingParameters, update_fiber_orientation, update_structural_order
from .stimuli import (
    directional_stretch_stimulus,
    equilibrium_structural_order,
    hill_activation,
    principal_stretch_direction,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration of the homogeneous remodeling demonstration."""

    total_time: float = 40.0
    dt: float = 0.05
    ramp_duration: float = 8.0
    maximum_stretch: float = 1.25
    initial_fiber_angle_deg: float = 60.0
    initial_beta: float = 0.1

    def __post_init__(self) -> None:
        if self.total_time <= 0.0 or not np.isfinite(self.total_time):
            raise ValueError("total_time must be finite and strictly positive.")
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be finite and strictly positive.")
        if self.ramp_duration <= 0.0 or self.ramp_duration > self.total_time:
            raise ValueError("ramp_duration must lie in (0, total_time].")
        if self.maximum_stretch < 1.0 or not np.isfinite(self.maximum_stretch):
            raise ValueError("maximum_stretch must be finite and at least one.")
        if not 0.0 <= self.initial_beta <= 1.0:
            raise ValueError("initial_beta must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    time: FloatArray
    deformation_gradient: FloatArray
    fiber_direction: FloatArray
    fiber_angle_deg: FloatArray
    structural_order: FloatArray
    equilibrium_order: FloatArray
    stimulus: FloatArray
    strain_energy: FloatArray
    cauchy_stress: FloatArray


def run_homogeneous_remodeling(
    config: SimulationConfig = SimulationConfig(),
    material: MaterialParameters = MaterialParameters(),
    remodeling: RemodelingParameters = RemodelingParameters(),
) -> SimulationResult:
    """Run a prescribed-load homogeneous remodeling simulation.

    Version 0.1 deliberately separates constitutive/remodeling verification from
    a boundary-value solver. The deformation gradient is prescribed; stress and
    internal variables are then evaluated consistently at every time step.
    """
    number_of_steps = int(np.floor(config.total_time / config.dt)) + 1
    time = np.arange(number_of_steps, dtype=float) * config.dt

    deformation_history = np.empty((number_of_steps, 2, 2), dtype=float)
    fiber_history = np.empty((number_of_steps, 2), dtype=float)
    beta_history = np.empty(number_of_steps, dtype=float)
    beta_equilibrium_history = np.empty(number_of_steps, dtype=float)
    stimulus_history = np.empty(number_of_steps, dtype=float)
    energy_history = np.empty(number_of_steps, dtype=float)
    stress_history = np.empty((number_of_steps, 2, 2), dtype=float)

    fiber = angle_to_vector(np.deg2rad(config.initial_fiber_angle_deg))
    beta = float(config.initial_beta)

    for step, current_time in enumerate(time):
        deformation = area_preserving_uniaxial_deformation(
            current_time,
            maximum_stretch=config.maximum_stretch,
            ramp_duration=config.ramp_duration,
        )
        stimulus = float(directional_stretch_stimulus(deformation))
        beta_equilibrium = float(
            equilibrium_structural_order(
                stimulus,
                beta_min=remodeling.beta_min,
                beta_max=remodeling.beta_max,
                half_saturation=remodeling.half_saturation,
                hill_exponent=remodeling.hill_exponent,
            )
        )

        deformation_history[step] = deformation
        fiber_history[step] = fiber
        beta_history[step] = beta
        beta_equilibrium_history[step] = beta_equilibrium
        stimulus_history[step] = stimulus
        energy_history[step] = strain_energy_density(deformation, fiber, beta, material)
        stress_history[step] = cauchy_stress(deformation, fiber, beta, material)

        if step + 1 < number_of_steps:
            target_direction = principal_stretch_direction(deformation)
            orientation_activation = float(
                hill_activation(
                    stimulus,
                    half_saturation=remodeling.half_saturation,
                    hill_exponent=remodeling.hill_exponent,
                )
            )
            fiber = update_fiber_orientation(
                fiber,
                target_direction,
                rate=remodeling.orientation_rate * orientation_activation,
                dt=config.dt,
            )
            beta = float(
                update_structural_order(
                    beta,
                    beta_equilibrium,
                    rate=remodeling.order_rate,
                    dt=config.dt,
                )
            )

    return SimulationResult(
        time=time,
        deformation_gradient=deformation_history,
        fiber_direction=fiber_history,
        fiber_angle_deg=np.rad2deg(vector_to_angle(fiber_history)),
        structural_order=beta_history,
        equilibrium_order=beta_equilibrium_history,
        stimulus=stimulus_history,
        strain_energy=energy_history,
        cauchy_stress=stress_history,
    )
