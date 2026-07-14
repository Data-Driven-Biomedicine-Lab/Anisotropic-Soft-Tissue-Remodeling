# Mathematical model used in version 0.3

## Scope

Version 0.3 is a constitutive, remodeling, and optical-preprocessing demonstrator under prescribed
kinematics. It includes both a homogeneous material-point example and a spatially
heterogeneous field example. It is not yet a finite-element boundary-value solver.
This separation makes the constitutive equations and evolution laws easy to verify
before mechanical equilibrium and spatial regularization are introduced.

## Kinematics

For the deformation gradient **F**, define

- `J = det(F) > 0`,
- `C = F^T F`,
- `I1 = tr(C)`,
- `I4 = a0 · C a0`,

where `a0` is the unit fiber orientation in the reference configuration.

## Strain-energy density

The matrix contribution is compressible neo-Hookean:

`psi_m = mu/2 (I1 - 2 - 2 ln J) + kappa/2 (ln J)^2`.

The tension-only fiber contribution is

`psi_f = beta k1/(2 k2) [exp(k2 <I4 - 1>_+^2) - 1]`.

Thus `psi = psi_m + psi_f`. The scalar `beta in [0, 1]` represents structural
order and scales the effective fiber contribution.

The first Piola stress is obtained analytically as `P = d psi / d F`, and the
Cauchy stress is `sigma = J^(-1) P F^T`.

## Remodeling stimulus

The directional stretch stimulus is

`S = |ln(lambda_max) - ln(lambda_min)|`,

where `lambda_max` and `lambda_min` are the principal stretches. This stimulus
is objective, non-negative, and zero for isotropic stretches.

## Structural-order evolution

The equilibrium structural order is represented by a bounded Hill law:

`beta_eq(S) = beta_min + (beta_max - beta_min) S^n / (S_half^n + S^n)`.

The kinetic law is

`beta_dot = k_beta (beta_eq - beta)`.

The code uses its exact one-step update for a fixed `beta_eq`.

## Orientation evolution

The fiber angle is treated as a nematic quantity: angles separated by `pi` are
equivalent. At every step, the target direction is the eigenvector of `C`
associated with the largest principal stretch. The current angle relaxes toward
the target through the shortest nematic angular difference, using an exponential
one-step update. The orientation rate is multiplied by the same Hill activation
used for the structural response, so no preferred direction is imposed when the
stretch state is isotropic.

## Compatible spatial example

For a reference rectangle with half-height `H`, the spatial demonstration uses

`x = lambda X - gamma H/pi cos(pi Y/H)`,

`y = Y/lambda`.

The resulting deformation gradient is

`F = [[lambda, gamma sin(pi Y/H)], [0, 1/lambda]]`,

and satisfies `det(F) = 1` pointwise. The fields `a0(X,Y,t)` and `beta(X,Y,t)`
are evolved locally with vectorized versions of the same constitutive and kinetic
relations used in the homogeneous example. No spatial derivatives or force-balance
solve are included at this stage.


## Polarimetry-to-structure reconstruction

The optical preprocessing stage accepts a fast-axis azimuth map `alpha(X,Y)`
and a retardance map `R(X,Y)`. The azimuth is canonicalized modulo `pi`, because
a fiber direction is head-tail symmetric. The model-ready orientation and
structural tensor are

`a0 = [cos(alpha), sin(alpha)]`,

`A = a0 tensor a0`.

Retardance is converted to a bounded structural-order proxy through an explicit
calibration:

`R_hat = clip((R - R_min)/(R_max - R_min), 0, 1)`,

`beta_R = beta_min + (beta_max - beta_min) R_hat^p`.

Local nematic coherence is computed with doubled angles:

`q = <w exp(2 i alpha)>/<w>`, `c = |q|`.

The default combined proxy is `beta = beta_R c`. This relation is a documented
preprocessing assumption, not a universal identity between retardance and
mechanical anisotropy. Quantitative interpretation requires an explicitly stated
calibration for thickness, wavelength, staining, optical setup, and independent
structural or mechanical measurements.

Invalid pixels are excluded using finite-value checks, a minimum-retardance
threshold, and an optional external segmentation or quality mask. Floating-point
outputs at invalid pixels are represented by `NaN` and accompanied by a Boolean
validity mask.

## Finite-element equilibrium and staggered remodeling

The mechanically coupled model uses a total-Lagrangian Q4 discretization. For a
fixed element structure, the displacement field satisfies the weak equilibrium
condition

`integral P : grad(delta u) dV = 0`.

After equilibrium is obtained, element-averaged deformation gradients determine
the principal-stretch targets and directional stimuli. Fiber orientation and
structural order are advanced over one remodeling interval, and equilibrium is
solved again with the previous displacement field as a warm start. The applied
boundary displacement is held fixed, so changes in reaction force arise from the
evolving material structure and the corresponding redistribution of deformation.

## Interpretation and limitations

The laws are deliberately minimal and phenomenological. They provide a
reproducible computational baseline, not a universal biological law. The current
finite-element model is two-dimensional, has one fiber family per element, uses
local staggered kinetics without spatial regularization, and is demonstrated on
synthetic polarimetry. Future versions will add richer fiber architecture,
regularization, parameter identification, uncertainty analysis, and blind
synthetic validation on held-out loading protocols.


## Multiple fiber families

Version 0.7 replaces the single fiber contribution by a normalized
mixture of discrete families:

\[
\psi_f =
\sum_{m=1}^{M}
w_m\beta_m\frac{k_{1m}}{2k_{2m}}
\left[
\exp\left(k_{2m}\langle I_{4m}-1\rangle_+^2\right)-1
\right].
\]

The first Piola stress is obtained analytically by summing the
family-wise derivatives. A continuous orientation distribution is
represented by sufficiently many quadrature directions and normalized
weights.

## Spatial regularization

Optional regularization solves an implicit graph-Laplacian problem on
the finite-element adjacency graph. Structural-order fields are
regularized directly. Nematic directions are regularized through
\((\cos 2\alpha,\sin 2\alpha)\), then normalized and mapped back to
\(\alpha\).


## Parameter identification

Version 0.8 estimates positive constitutive parameters in logarithmic
coordinates by minimizing the weighted least-squares objective

\[
\Phi(\mathbf p)
=
\sum_i
\left[
\frac{
\sigma_i^{\mathrm{model}}(\mathbf p)
-
\sigma_i^{\mathrm{obs}}
}{s_i}
\right]^2.
\]

Local sensitivities are evaluated by centered finite differences. The
singular values of the weighted log-parameter sensitivity matrix are
used to diagnose local identifiability.

Under strictly isochoric loading, \(J=1\) and the volumetric term
vanishes. Therefore, \(\kappa\) is structurally unidentifiable unless
the dataset contains a non-isochoric protocol.
