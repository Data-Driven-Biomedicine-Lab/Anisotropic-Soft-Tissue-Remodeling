# Finite-element equilibrium

Version 0.5 introduces the first spatial boundary-value solver in the repository.
It uses a total-Lagrangian formulation with four-node bilinear quadrilaterals,
2x2 Gauss integration, the existing finite-strain anisotropic constitutive law,
and incremental displacement control.

## Strong and weak forms

The equilibrium equation in the reference configuration is

\[
\operatorname{Div}_{X}\mathbf P=\mathbf 0,
\qquad
\mathbf P=\partial\psi/\partial\mathbf F,
\qquad
\mathbf F=\mathbf I+\nabla_X\mathbf u.
\]

The discrete solution minimizes the internal strain energy under essential
boundary conditions. The left and right edges receive prescribed horizontal
displacements; one vertical degree of freedom removes rigid translation, while
all remaining vertical degrees of freedom are traction-free.

## Image-to-mesh transfer

Polarimetric structural fields are sampled at element centroids. Fiber
orientation is averaged in doubled-angle space, preserving the nematic
invariance \(\mathbf a\equiv-\mathbf a\). Structural order is interpolated with
the same inverse-distance weights.

## Numerical checks

The executable notebook verifies:

- convergence at every load increment;
- a small residual on unconstrained degrees of freedom;
- positive current Jacobians;
- balance of edge reactions;
- symmetry of the Cauchy stress;
- lower energy than an admissible affine comparison field.

## Current limitation

Structure is fixed during the equilibrium solve. Version 0.6 will alternate
between equilibrium, stimulus evaluation, and structural updates.
