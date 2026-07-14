# Public API overview

The package is organized into independent layers so that constitutive laws,
image-like structural reconstruction, finite-element equilibrium, remodeling,
and inverse problems can be tested separately.

## Core mechanics

- `MaterialParameters`
- `strain_energy_density`
- `first_piola_stress`
- `cauchy_stress`
- `MultiFiberMaterialParameters`
- `multifiber_strain_energy_density`
- `multifiber_first_piola_stress`
- `multifiber_cauchy_stress`

## Orientation and structure

- `angle_to_vector`
- `vector_to_angle`
- `orientation_tensor`
- `nematic_angle_difference`
- `regularize_nematic_field`
- `regularize_scalar_field`

## Synthetic polarimetry-like reconstruction

- `RetardanceCalibration`
- `retardance_to_order_proxy`
- `local_nematic_coherence`
- `polarimetry_to_structure`
- `synthetic_polarimetry_benchmark`

## Remodeling

- `RemodelingParameters`
- `update_fiber_orientation`
- `update_structural_order`
- `run_homogeneous_remodeling`
- `run_spatial_remodeling`
- `run_equilibrium_remodeling`

## Finite elements

- `rectangular_quad_mesh`
- `solve_displacement_controlled_equilibrium`
- `solve_multifiber_displacement_controlled_equilibrium`
- `sample_nematic_image_to_elements`

## Identification and validation

- `MechanicalDataset`
- `MaterialParameterMap`
- `fit_material_parameters`
- `local_sensitivity_matrix`
- `parametric_bootstrap_material_fit`
- `create_synthetic_validation_challenge`
- `evaluate_synthetic_challenge`

The notebook series provides complete usage examples and numerical checks for
all major layers.
