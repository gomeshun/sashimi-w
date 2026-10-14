# Reproducing the c=2.5/shared-top-hat references

The frozen author source is upstream/main at
`99dfc3632eec0126080c0273ddf78f84fe09216c`. Obtain that Git object in a
SASHIMI-W checkout; the controller exports that object, never its working tree.
The immutable controller and worker are copied byte-for-byte from
[sashimi-family e8fc3f5](https://github.com/gomeshun/sashimi-family/tree/e8fc3f5b038c8366a2e6ef99b269f3cf42fbe7f2/scripts).
They are preserved evidence; do not edit them. Sidecars record both hashes.
The inherited patches are also byte-for-byte copies. Patch 11 independently
implements the new mass assignment and amplitude. Optional patch 12 supplies
explicit ODE controls and changes no physical rate.

The controller runs under Python 3.11 or newer. Supply a separate reference
interpreter with Python 3.10.20, NumPy 1.22.4, SciPy 1.13.1,
Matplotlib 3.7.5 and mpmath 1.3.0. For example, from this repository root:

```bash
python tests/reference-generation/historical-controller/run_reference.py \
  --repository . \
  --config tests/reference-generation/configs/top-hat-c2.5-2.0keV-refined.json \
  --python /absolute/path/to/reference-venv/bin/python \
  --output /tmp/top-hat-c2.5-2.0keV-refined.npz
```

Use the un-suffixed configurations for standalone historical-default ODE
regression at 0.5, 2 and 5 keV. Migration's strict 2 keV regression uses
`-refined`; its numerical convergence check uses `-refined-check`. These
configurations and their source hashes distinguish numerical controls from
physical prescriptions. The 5 keV refined run is a known diagnostic limitation,
not an accepted precision reference. See `docs/w-power-prescription.md`.

`ablation.py` is a new active diagnostic tool. Run old/filter variants with
PYTHONPATH set to an export of owner main 685bcfde7bf3ac5805492b56070151ba0415a9ca;
run normalization/both variants with the changed standalone source. Every
run accepts `--variant` and `--output`. Only the diagnostic cutoff is replaced;
normalization is held fixed for the filter-only comparison. The records under
`docs/validation/w-prescription-20261014` identify the settings and source hashes.
