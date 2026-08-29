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

```bash
pip install -r requirements.txt
python -m scripts.synthetic_workflow
```

The real-data workflow and plotting utilities are also available as modules:

```bash
python -m scripts.real_workflow
python -m scripts.plot_results
python -m scripts.coherent_sum
```

Generated files are kept under `outputs/`, so the project root stays focused on code and configuration.
