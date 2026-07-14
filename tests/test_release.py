import json
from pathlib import Path

from anisotropic_remodeling import __version__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_stable() -> None:
    assert __version__ == "1.0.0"


def test_release_manifest_is_synthetic_and_complete() -> None:
    manifest = json.loads(
        (REPOSITORY_ROOT / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == __version__
    assert manifest["data_origin"] == "fully synthetic"
    assert len(manifest["files"]) >= 7
    for relative, metadata in manifest["files"].items():
        assert (REPOSITORY_ROOT / relative).is_file()
        assert len(metadata["sha256"]) == 64
        assert metadata["size_bytes"] > 0


def test_notebook_series_is_complete() -> None:
    notebook_directory = REPOSITORY_ROOT / "notebooks"
    expected = [
        f"{index:02d}_{name}.ipynb"
        for index, name in (
            (1, "homogeneous_remodeling"),
            (2, "spatial_fiber_field"),
            (3, "polarimetry_to_structure"),
            (4, "polarimetry_initialized_remodeling"),
            (5, "finite_element_equilibrium"),
            (6, "equilibrium_remodeling_coupling"),
            (7, "multiple_fiber_families_and_regularization"),
            (8, "parameter_identification_and_sensitivity"),
            (9, "synthetic_validation_challenge"),
            (10, "release_reproducibility_audit"),
        )
    ]
    for filename in expected:
        assert (notebook_directory / filename).is_file()
