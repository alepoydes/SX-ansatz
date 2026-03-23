# SX-ansatz

This repository contains two `marimo` notebooks reproducing the `S1`, `S2`, and `S3` toron ansatzes discussed in the article ["Optical Imaging and Analytical Design of Localized Topological Structures in Chiral Liquid Crystals"](https://www.mdpi.com/2073-8994/14/12/2476).

The notebooks describe the same objects in two different ways:

- `notebooks/article_torons.py` writes the ansatzes in a maximally explicit form, closely following the construction described in the article.
- `notebooks/s1_s2_s3_ansatz.py` uses TopoVec's curvilinear-coordinate machinery to express the same ansatzes in a more abstract form.
- Both notebooks are intended to produce the same `S1`/`S2`/`S3` director fields and the same rendered slice panels.

## Prerequisites

Install the following tools first:

- `git`
- `uv` (installation instructions: [official docs](https://docs.astral.sh/uv/getting-started/installation/))

## Clone the repository

Once this repository is published on GitHub, clone it and enter the project directory:

```sh
git clone https://github.com/alepoydes/SX-ansatz.git
cd SX-ansatz
```

## Install dependencies

Create the project environment and install all notebook dependencies:

```sh
uv sync
```

## Run the notebooks

Launch the notebook that follows the article formulation directly:

```sh
uv run marimo edit notebooks/article_torons.py
```

Launch the notebook that uses the curvilinear-coordinate TopoVec ansatz machinery:

```sh
uv run marimo edit notebooks/s1_s2_s3_ansatz.py
```

The notebooks expect a writable `tmp/` directory at the repository root. It is already included in this project.

## Deep inspection inside the notebooks

For deeper interactive inspection of a computed director field, use TopoVec's marimo inspector helper (currenly commented):

```python
tv.marimo.inspect(s1_nn)
```


Typical workflow:

1. Open one of the notebooks with `marimo`.
2. Run the cells until `s1_nn`, `s2_nn`, or `s3_nn` has been computed.
3. Add a new cell with `tv.marimo.inspect(...)`, or uncomment the existing example cell in `notebooks/s1_s2_s3_ansatz.py`.
4. Re-run that cell to open the live TopoVec inspector widget inside the notebook.

## Inspect exported states in the desktop viewer

If you want to inspect a saved state outside `marimo`, export it to `.npz` and open it with the TopoVec desktop viewer.

The notebooks already contain commented `tv.io.save_npz_lcsim(...)` examples next to the generated states. Uncomment the relevant line, run that cell, and then open the result:

```sh
uv run topovec view tmp/toron_S1_0.5um.npz
```

or, for the explicit article notebook output:

```sh
uv run topovec view tmp/article_toron_S1_0.5um.npz
```

The exact filename depends on the selected notebook and grid step.
