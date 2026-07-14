# Polarimetry-to-structure pipeline

Version 0.3 converts two-dimensional azimuth and retardance maps into initial
structural fields for the remodeling model:

\[
\alpha(X,Y), R(X,Y)
\rightarrow
\mathbf a_0(X,Y), \beta(X,Y),
\mathbf A(X,Y)=\mathbf a_0\otimes\mathbf a_0.
\]

## Input conventions

- `azimuth_rad`: synthetic fast-axis azimuth-like field in radians;
- `retardance`: finite retardance signal in documented units;
- `external_valid_mask`: optional segmentation or quality mask;
- `orientation_offset_rad`: prescribed synthetic offset between the optical-like
  axis and the modeled fiber axis.

Azimuth is treated as nematic: `alpha` and `alpha + pi` represent the same
orientation.

## Retardance calibration

Retardance is mapped to a bounded structural-order proxy using
`RetardanceCalibration`. The mapping is monotone, clipped, and explicit. It is
not a universal physical identity between phase delay and tissue anisotropy.
Quantitative biological interpretation requires calibration for specimen
thickness, wavelength, staining protocol, optical setup, and an independent
structural or mechanical reference.

## Local coherence

The local orientation coherence is calculated with doubled angles:

\[
q=\frac{\langle w\exp(2i\alpha)\rangle}{\langle w\rangle},
\qquad c=|q|.
\]

The default structural proxy is `beta = beta_R * c`, where `beta_R` is the
retardance-derived proxy. Set `combine_with_coherence=False` to retain
`beta_R` alone.

## Invalid data

Pixels are invalid when azimuth or retardance is non-finite, retardance falls
below the requested threshold, or the external mask is false. Floating-point
outputs at invalid pixels are set to `NaN`; `valid_mask` is always returned.

## Outputs

- canonical azimuth in `[0, pi)`;
- unit fiber direction `a0`;
- structural tensor `A = a0 tensor a0`;
- retardance-order proxy;
- local nematic coherence;
- combined structural-order proxy;
- validity mask.

See `notebooks/03_polarimetry_to_structure.ipynb` for a complete executable
example and validation against synthetic ground truth.
