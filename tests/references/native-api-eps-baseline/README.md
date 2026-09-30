# Corrected EPS references for native integration

Generated with `scripts/generate_corrected_native_reference.py` from separately
built and installed migration 65f25390f7d745d671818ba904c59b133ce1aa29 and ITAMAE
23d01e8758a88b061b87de9e488c38ec89fd8e4f, without importing the native API.

- `sigma_offset`: same constructor/grid as the historical six-case fixture;
  named columns and weights use the corrected exact-z=1 deterministic anchor
- `deterministic_rate_rows`: original nonsquare mass/redshift inputs, with each
  one-dimensional row independently evaluated by the corrected migration

JSON sidecars record source provenance and SHA256. Old fixtures are untouched.
CI also generates all six corrected catalogs and row rates on each runner and
requires exact array equality, separately from saved-file transport checks.
