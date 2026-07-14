# Verification status for version 0.1

The reference implementation was checked with automated tests and static analysis.

## Automated tests

The test suite verifies:

- nematic angle/vector conversions and sign invariance;
- unit trace of the structural tensor;
- zero energy and zero stress in the reference state;
- tension-only fiber response;
- symmetry of the Cauchy stress;
- positivity of `det(F)`;
- agreement of the analytical first Piola stress with a centered finite-difference
  derivative of the strain-energy density;
- principal-stretch direction and directional-stimulus behavior;
- bounded monotone Hill activation;
- stable orientation and structural-order updates;
- successful execution of the complete reference simulation.

Result at packaging time: **16 tests passed**.

## Static checks

- `ruff check .`: passed;
- `python -m compileall`: passed;
- reference example: executed successfully;
- output figure: generated successfully.
