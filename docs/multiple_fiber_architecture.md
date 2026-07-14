# Multiple fiber families and spatial regularization

Version 0.7 extends the single-family model in two complementary directions.

## Discrete fiber mixture

At each material point, the tissue may contain \(M\) undirected fiber families
\(\mathbf a_0^{(m)}\), structural-order values \(\beta_m\), and normalized
mixture weights \(w_m\). The total energy is

\[
\psi = \psi_m +
\sum_{m=1}^{M}
w_m \beta_m
\frac{k_{1m}}{2k_{2m}}
\left[
\exp\left(
k_{2m}\langle I_{4m}-1\rangle_+^2
\right)-1
\right],
\]

where

\[
I_{4m} =
\mathbf a_0^{(m)}
\cdot
\mathbf C
\mathbf a_0^{(m)}.
\]

Every family remains tension-only and nematic:
\(\mathbf a_0^{(m)}\equiv-\mathbf a_0^{(m)}\).

## Orientation distribution

A continuous two-dimensional orientation distribution can be approximated by
many discrete quadrature directions. The supplied benchmark uses the
pi-periodic von Mises density

\[
\rho(\theta)
=
\frac{\exp[\kappa\cos 2(\theta-\bar\theta)]}
{\pi I_0(\kappa)},
\qquad
0\leq\theta<\pi.
\]

The concentration parameter \(\kappa\) controls dispersion. Its theoretical
nematic coherence is \(I_1(\kappa)/I_0(\kappa)\).

## Graph regularization

Image-derived element fields often contain pixel-scale noise. Optional
regularization uses the element-adjacency graph with Laplacian \(\mathbf L\).
For a scalar field \(\mathbf y\), the regularized field solves

\[
(\mathbf I+\ell\mathbf L)\mathbf x=\mathbf y.
\]

Fiber directions are not smoothed componentwise. They are mapped to doubled
angle space,

\[
\mathbf q=(\cos 2\alpha,\sin 2\alpha),
\]

regularized there, normalized, and mapped back. This preserves head-tail
symmetry and avoids cancellation between equivalent directions separated by
\(180^\circ\).

## Interpretation and limitations

Regularization is an explicit modeling choice, not a substitute for better
measurements. The dimensionless strength \(\ell\) depends on mesh resolution
and should later be selected by parameter identification or cross-validation.
The current model treats families as discrete contributions and does not yet
include family-specific turnover kinetics or a fully continuous orientation
distribution inside the remodeling loop.
