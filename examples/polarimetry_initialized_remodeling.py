"""Run the polarimetry-initialized remodeling demonstration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anisotropic_remodeling import (
    MaterialParameters,
    PolarimetryRemodelingConfig,
    RemodelingParameters,
    RetardanceCalibration,
    polarimetry_to_structure,
    run_polarimetry_initialized_remodeling,
)


repository_root = Path(__file__).resolve().parents[1]
raw = np.load(repository_root / "data" / "synthetic" / "polarimetry_maps.npz")

calibration = RetardanceCalibration(
    lower_retardance=0.04,
    upper_retardance=0.90,
    beta_min=0.05,
    beta_max=0.95,
    exponent=1.30,
)
structure = polarimetry_to_structure(
    raw["azimuth_rad"],
    raw["retardance"],
    calibration,
    minimum_valid_retardance=0.08,
    external_valid_mask=raw["tissue_mask"],
    coherence_window=9,
)

config = PolarimetryRemodelingConfig(
    total_time=12.0,
    dt=0.1,
    ramp_duration=4.0,
    maximum_stretch=1.18,
    maximum_shear=0.35,
    half_height=float(np.max(np.abs(raw["y"]))),
)
result = run_polarimetry_initialized_remodeling(
    raw["x"],
    raw["y"],
    structure.fiber_direction,
    structure.structural_order,
    structure.valid_mask,
    config,
    MaterialParameters(),
    RemodelingParameters(),
    snapshot_times=(0.0, config.ramp_duration, config.total_time),
)

figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
for index, time in enumerate(result.snapshot_time):
    image = axes[0, index].imshow(
        result.structural_order[index],
        origin="lower",
        extent=(raw["x"].min(), raw["x"].max(), raw["y"].min(), raw["y"].max()),
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axes[0, index].set(title=f"beta, t={time:g}", xlabel="X", ylabel="Y")
    figure.colorbar(image, ax=axes[0, index])

axes[1, 0].plot(result.time, result.mean_structural_order)
axes[1, 0].set(title="Mean structural order", xlabel="Time", ylabel="Mean beta")
axes[1, 1].plot(result.time, result.mean_cauchy_stress_xx)
axes[1, 1].set(title="Mean axial stress", xlabel="Time", ylabel="Mean sigma_xx")
axes[1, 2].plot(result.time, result.mean_target_alignment)
axes[1, 2].set(title="Mean target alignment", xlabel="Time", ylabel="Alignment")
for axis in axes[1]:
    axis.grid(True, alpha=0.25)

output_path = repository_root / "results" / "figures" / "polarimetry_initialized_remodeling.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path, dpi=180)
print(f"Saved figure to {output_path.relative_to(repository_root)}")
