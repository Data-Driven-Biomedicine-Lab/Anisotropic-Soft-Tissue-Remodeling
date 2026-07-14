# Staggered finite-element equilibrium-remodeling coupling

Version 0.6 closes the first mechanically coupled remodeling loop. The applied
boundary displacement is held fixed, but the displacement field is not
prescribed inside the sample. At every remodeling time point, mechanical
equilibrium is recomputed for the current element structure.

## Algorithm

For element fields \(\mathbf a_0^e(t)\) and \(\beta^e(t)\):

1. solve the nonlinear finite-element equilibrium problem;
2. extract the element-averaged deformation gradient \(\mathbf F^e\);
3. compute the principal-stretch target direction \(\mathbf n_{\max}^e\);
4. compute the directional stimulus
   \[
   S^e = |\ln\lambda_{\max}^e-\ln\lambda_{\min}^e|;
   \]
5. update the nematic fiber direction and bounded structural order;
6. use the previous displacement field as the warm start for the next equilibrium solve.

The kinetic updates remain exact over a time interval when their target values
are held fixed. The final interval is shortened automatically when `total_time`
is not an integer multiple of `dt`.

## Interpretation

Under fixed displacement, remodeling changes both the local stress distribution
and the boundary reaction. This is the first version in which structure changes
mechanics and the updated mechanics feeds back into subsequent structural
change.

## Synthetic benchmark status

The executable example uses deterministic synthetic polarimetry because measured
maps are not currently available. The benchmark contains known latent fields,
measurement noise, and a low-signal defect. It verifies the software pipeline;
it does not constitute experimental validation.

## Current limitations

- one fiber family per element;
- local kinetic laws without nonlocal regularization;
- fixed applied displacement rather than physiological loading;
- no specimen-specific structural or mechanical calibration;
- two-dimensional plane model;
- staggered rather than monolithic coupling.
