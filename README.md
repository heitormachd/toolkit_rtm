# Acoustic Imaging Toolkit

GPU-accelerated acoustic/seismic imaging with WebGPU compute shaders. The repository contains real-data and synthetic workflows, time reversal, standard RTM, and Poynting-vector RTM.

For the full architecture, data formats, and workflow details, see [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

## Repository layout

- `acoustics_imaging/` — reusable simulation and processing code
- `scripts/` — runnable workflows and analysis utilities
- `shaders/` — WGSL compute kernels
- `assets/models/` — color-coded velocity/input images
- `assets/sources/` — source waveforms (`source.npy`, `source0.npy`, ...)
- `data/` — local recorded datasets such as Acude and Panther data
- `outputs/` — generated simulation and analysis results
- `tests/` — automated tests

## Quick start

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync
uv run python -m scripts.synthetic_workflow
```

Set `ENABLE_POYNTING_VECTORS = False` in `scripts/synthetic_workflow.py` to run
only conventional RTM.

The synthetic workflow uses sparse full matrix capture by default: every 16th
receiver transmits once and every receiver records each shot. Set
`ENABLE_SPARSE_FMC = False` for the original bitmap-source single shot, or set
`FMC_TRANSMITTER_STRIDE = 1` for full (non-sparse) capture.

The real-data workflow and plotting utilities are also available as modules:

```bash
uv run python -m scripts.real_workflow
uv run python -m scripts.plot_results
uv run python -m scripts.coherent_sum
```

Generated files are kept under `outputs/`, so the project root stays focused on code and configuration.
