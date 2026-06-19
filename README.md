# NNEoS — Neural Network Equation of State

Generates thermodynamic EOS tables usable in hybrid models for heavy-ion collisions
at √s_NN = 2–100 GeV, based on [arXiv:2605.22199](https://arxiv.org/abs/2605.22199).
For each point on an (ε, n_B) grid, the code solves for (T, μ_B, μ_Q, μ_S)
using damped Newton iterations.
Ready-to-use tables can be found in the [`data/`](data/) directory.

## Requirements

- Python 3.10
- See `requirements.txt`

## Installation

```bash
git clone https://github.com/Musfer/NNEoS.git
cd NNEoS
python -m venv NNEoS_venv
source NNEoS_venv/bin/activate  # On Windows: NNEoS_venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Open and run the notebook:

```bash
jupyter lab generateTable.ipynb
```

Define your (ε, n_B) grid in the marked cell, then run all cells.

## Output

`ExampleEOS.dat` — columns: `T [GeV]  muB [GeV]  ε [GeV/fm^3]  nB [1/fm^3]  muQ [GeV]  muS [GeV]  P [GeV/fm^3]  residual`

The files `smallEOS.dat` and `bigEOS.dat` are pregenerated tables on two different grid sizes, ready for direct use:
- `smallEOS.dat` — near-particlization region: ε ∈ [0.00, 0.80] GeV/fm³, n_B ∈ [-0.10, 0.80] fm⁻³, step 0.005
- `bigEOS.dat` — hydrodynamic region: ε ∈ [0.00, 80.0] GeV/fm³, n_B ∈ [-0.50, 8.00] fm⁻³, steps 0.1 and 0.05 respectively

The residual column is the RMS of the Newton residual vector (P, ε, n_B, n_Q, n_S components, without dimensional normalization).
Large residuals (typically close to 1.0) indicate that the (ε, n_B) combination is unphysical and cannot be realised by this EOS.

## Compatibility with vHLLE (eoChiral.cpp)

The output `.dat` files are compatible with the chiral EOS reader impelemented in vHLLE:
https://github.com/yukarpenko/vhlle/blob/main/src/eoChiral.cpp

When integrating, note the following differences from the default reader:

**Table format**
1. Remove the ε and n_B rescaling present in `eoChiral.cpp` — the values in
   these files are already in physical units.
2. Remove any dimensional rescaling — all quantities are already in GeV and fm.
3. Do not override μ_Q to 0.0 — the table contains the physical μ_Q values.

**Hybrid model usage remarks**
1. No switching to a hadronic EOS is needed: this model smoothly connects to
   hadronic degrees of freedom for ε ≤ 0.5 GeV/fm³, which is the typical
   particlization criterion used in hybrid simulations.
2. When generating particles in the hadron sampler, do not use the
   Boltzmann series decomposition (typically applied when μ_B/T is small).

## Citation

If you use this EOS in your work, please cite the following papers:

**This work** (NNEoS model description):
> M. Adzhymambetov, *Equation of State at High Baryon Densities from a Thermodynamically Informed Neural Network*, arXiv:2605.22199 [hep-ph] (2026).
> https://arxiv.org/abs/2605.22199 (or published version if available)

**This code** (NNEoS repository):
> M. Adzhymambetov, *NNEoS: Neural Network Equation of State*, Zenodo (2026).
> https://doi.org/10.5281/zenodo.20763330

**Lattice QCD EOS baseline** (the parametrization this work builds upon):
> J. Noronha-Hostler et al., *QCD equation of state at finite chemical potentials for use in hydrodynamic simulations*, arXiv:1902.06723 (2019).
> https://arxiv.org/abs/1902.06723
> https://doi.org/10.1103/PhysRevC.100.064910

**Lattice QCD simulations** (underlying lattice data):
> R. Bellwied et al., *Fluctuations and correlations of net baryon number, electric charge, and strangeness*, arXiv:1805.04445 (2018).
> https://arxiv.org/abs/1805.04445   
> https://doi.org/10.1007/JHEP10%282018%29205
