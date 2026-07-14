"""Plotting helpers for the reference simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .simulation import SimulationResult


def plot_simulation_summary(
    result: SimulationResult,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot loading, orientation, order and axial Cauchy stress histories."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    axial_stretch = result.deformation_gradient[:, 0, 0]
    axes[0, 0].plot(result.time, axial_stretch)
    axes[0, 0].set(xlabel="Time", ylabel="Axial stretch", title="Prescribed loading")

    axes[0, 1].plot(result.time, result.fiber_angle_deg)
    axes[0, 1].set(xlabel="Time", ylabel="Fiber angle [deg]", title="Orientation remodeling")

    axes[1, 0].plot(result.time, result.structural_order, label=r"$\beta$")
    axes[1, 0].plot(
        result.time,
        result.equilibrium_order,
        linestyle="--",
        label=r"$\beta_{eq}$",
    )
    axes[1, 0].set(xlabel="Time", ylabel="Structural order", title="Order remodeling")
    axes[1, 0].legend()

    axes[1, 1].plot(result.time, result.cauchy_stress[:, 0, 0])
    axes[1, 1].set(xlabel="Time", ylabel=r"$\sigma_{xx}$", title="Axial Cauchy stress")

    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200)
    if show:
        plt.show()
    return figure
