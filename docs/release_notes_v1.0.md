# Version 1.0.0 release notes

Version 1.0.0 is the first stable scientific-software release of **Anisotropic
Soft Tissue Remodeling**.

## Stable capabilities

- finite-strain matrix and tension-only fiber constitutive laws;
- analytical first Piola and Cauchy stresses;
- nematic orientation handling;
- bounded structural-order evolution;
- homogeneous and spatial remodeling;
- synthetic polarimetry-like reconstruction;
- total-Lagrangian Q4 finite elements;
- equilibrium-remodeling coupling;
- multiple fiber families and orientation distributions;
- graph-Laplacian regularization;
- weighted parameter identification and uncertainty analysis;
- blind held-out synthetic validation;
- deterministic benchmark exports and checksums.

## Evidence status

All distributed data and results are synthetic. The release is verified and
synthetic-validation-ready, but it does not claim experimental or clinical
validation.

## Compatibility

The public API documented in `docs/api_overview.md` is considered stable for
the 1.x series. Backward-incompatible changes should be reserved for a future
major release.
