# Polarimetry-initialized remodeling

Version 0.4 connects reconstructed polarimetric structure to the prescribed-kinematics remodeling model. The input fields are

- reference coordinates `x`, `y`;
- a head-tail symmetric unit direction `a0(X,Y)`;
- a bounded structural-order proxy `beta(X,Y)`;
- an explicit validity mask.

The simulation evolves only valid pixels. Invalid pixels are not silently interpolated and are written as `NaN` in floating-point output fields. This choice is consistent with the current locally uncoupled material-point model. A future equilibrium-based finite-element model will require a tissue-domain mesh and an explicit strategy for holes, boundaries, and missing measurements.

At every valid material point, the workflow is

```text
polarimetric maps
    -> calibrated structural fields
    -> constitutive response
    -> directional stretch stimulus
    -> fiber and structural-order evolution
```

The deformation field remains prescribed and compatible:

```text
F = [[lambda, gamma sin(pi Y/H)], [0, 1/lambda]], det(F) = 1.
```

The image-derived initialization is therefore tested independently of a finite-element equilibrium solver. This separation keeps the assumptions auditable and provides a reproducible baseline for later data assimilation and parameter calibration.


## Synthetic oracle comparison

The research notebook also propagates the exact synthetic structural fields through the same remodeling model. The reconstructed and oracle simulations use the same validity mask, constitutive parameters, kinetic parameters, prescribed deformation, and time grid. Their difference therefore measures propagation of reconstruction error through this computational pipeline; it is not a validation of the biological evolution law.

The time grid always reaches `total_time` exactly. If the requested `dt` does not divide the interval, the implementation uses a shorter final step rather than truncating the simulation or stepping beyond the requested end time.
