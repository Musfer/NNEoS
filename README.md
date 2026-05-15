# NNEoS — Neural Network Equation of State

Generates thermodynamic EOS tables from a pretrained deep residual neural network,
using lattice QCD pressure as a baseline.

For each point on an (ε, n_B) grid, the code solves for (T, μ_B, μ_Q, μ_S)
using damped Newton iterations and writes the result to `ExampleEOS.dat`.

## Requirements

- Python 3.10
- See `requirements.txt`

## Installation

- git clone https://github.com/Musfer/NNEoS.git
- cd NNEoS
- python -m venv NNEoS_venv
- source NNEoS_venv/bin/activate  # On Windows: NNEoS_venv\Scripts\activate
- pip install -r requirements.txt

## Usage

Open and run the notebook:

```bash
jupyter lab generateTable.ipynb
```

Define your (ε, n_B) grid in the marked cell, then run all cells.

## Output

`ExampleEOS.dat` — columns: `T  muB  eps  nB  muQ  muS  P  residual` (all in GeV and fm units)

## Project structure
