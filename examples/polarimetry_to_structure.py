"""Convert the repository's synthetic polarimetry maps to structural fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anisotropic_remodeling import RetardanceCalibration, polarimetry_to_structure


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    input_path = repository_root / "data" / "synthetic" / "polarimetry_maps.npz"
    output_path = repository_root / "results" / "data" / "polarimetry_structure_example.npz"

    if not input_path.exists():
        raise FileNotFoundError(
            "Synthetic input maps are missing. Run notebook "
            "03_polarimetry_to_structure.ipynb first."
        )

    measurement = np.load(input_path)
    calibration = RetardanceCalibration(
        lower_retardance=0.04,
        upper_retardance=0.90,
        beta_min=0.05,
        beta_max=0.95,
        exponent=1.30,
    )
    result = polarimetry_to_structure(
        measurement["azimuth_rad"],
        measurement["retardance"],
        calibration,
        minimum_valid_retardance=0.08,
        external_valid_mask=measurement["tissue_mask"],
        coherence_window=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=measurement["x"],
        y=measurement["y"],
        valid_mask=result.valid_mask,
        azimuth_rad=result.azimuth_rad,
        fiber_direction=result.fiber_direction,
        structure_tensor=result.structure_tensor,
        retardance_order=result.retardance_order,
        local_coherence=result.local_coherence,
        structural_order=result.structural_order,
    )

    print(f"Valid pixels: {np.count_nonzero(result.valid_mask)}")
    print(f"Mean structural order: {np.nanmean(result.structural_order):.6f}")
    print(f"Saved: {output_path.relative_to(repository_root)}")


if __name__ == "__main__":
    main()
