"""Build the versioned SHA-256 release manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anisotropic_remodeling import __version__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INCLUDED = (
    "data/synthetic/polarimetry_maps.npz",
    "data/synthetic/validation_challenge_01/public/challenge_manifest.json",
    "data/synthetic/validation_challenge_01/public/training_observations.csv",
    "data/synthetic/validation_challenge_01/public/test_inputs.csv",
    "data/synthetic/validation_challenge_01/reference/hidden_test_targets.csv",
    "results/data/synthetic_validation_metrics.csv",
    "results/data/synthetic_validation_parameter_estimates.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = {
    "project": "Anisotropic Soft Tissue Remodeling",
    "version": __version__,
    "data_origin": "fully synthetic",
    "files": {
        relative: {
            "sha256": sha256(REPOSITORY_ROOT / relative),
            "size_bytes": (REPOSITORY_ROOT / relative).stat().st_size,
        }
        for relative in INCLUDED
    },
}

output = REPOSITORY_ROOT / "release_manifest.json"
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(output)
