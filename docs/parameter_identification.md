# Parameter identification, sensitivity, and uncertainty

Version 0.8 introduces a reproducible inverse-problem workflow for the
multi-family constitutive model.

## Dataset

Each observation contains:

- a prescribed deformation gradient \(\mathbf F_i\);
- one observed synthetic Cauchy-stress component;
- a known measurement standard deviation;
- a loading-protocol label;
- a scalar load coordinate used for plotting.

The default benchmark combines area-preserving extension in two directions,
simple shear, and a small volumetric dilation.

## Weighted least squares

Positive material parameters are fitted in logarithmic coordinates by
minimizing

\[
\Phi(\mathbf p)
=
\sum_{i=1}^{N}
\left[
\frac{
\sigma_i^{\mathrm{model}}(\mathbf p)
-
\sigma_i^{\mathrm{obs}}
}{
s_i
}
\right]^2,
\]

where \(s_i\) is the known observation standard deviation.

The identifiable vector is

\[
\mathbf p
=
(\mu,k_{11},\ldots,k_{1M},k_{21},\ldots,k_{2M},\kappa),
\]

with optional exclusion of \(\kappa\).

## Identifiability

For isochoric protocols, \(J=1\) and \(\ln J=0\). Consequently, the volumetric
penalty does not contribute to the stress and \(\kappa\) is structurally
unidentifiable. A volumetric protocol is required to produce non-zero
sensitivity to \(\kappa\).

The local sensitivity matrix is evaluated by centered finite differences.
Singular values and the condition number quantify local ill-conditioning.

## Uncertainty

Two complementary approximations are reported:

1. a local covariance matrix derived from the weighted least-squares Jacobian;
2. a deterministic parametric bootstrap in which noisy datasets are generated
   around the fitted prediction and refitted.

These intervals quantify uncertainty conditional on the selected model,
protocols, noise model, known fiber architecture, and parameter bounds. They do
not account for model-form discrepancy or uncertainty in the structural maps.
