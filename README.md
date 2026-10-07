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

Set `OUTPUT_SUBFOLDER_NAME` near the top of either workflow to keep a run in a
named subfolder, such as `poynting_good_result`. Use `None` or `''` to use the
generic output directory. Panther automatically uses each acquisition's complete
directory name instead. Reusing a name overwrites files from that run.

The real workflow defaults to both acquisitions in `panther_data/`. Start with
one transmitter from each recording (all receivers are used):

```bash
uv run python -m scripts.real_workflow --emitters 32
```

Omit `--emitters` to migrate every transmitter, use `--datasets meia_lua_fmc.m2k`
to select one acquisition, or add `--prepare-only` to check loading and model
setup without running WebGPU. Set `dados = 'acude'` for the original açude TR.
Outputs go to `outputs/simulations/real/<acquisition>.m2k/`, including
`ReverseTimeMigration/rtm_fmc.png`, the signed per-emitter images and FMC sum,
the velocity model, physical coordinate arrays, and `run_config.json`.

Standard/Poynting comparison is enabled by `ENABLE_POYNTING_VECTORS = True`.
`ReverseTimeMigration/fmc_comparison.png` shows both source-energy-normalized
images on the same colour scale, as in `poynting_good_result`. The workflow also
saves raw and normalized Poynting arrays and `fmc_transmitters.npy`. The 120°
angle condition uses particle velocity collocated with pressure at half time
steps; both comparison panels use those same pressure samples. Unit source
directions are checkpointed as float16; pressure remains float32. Use
`--no-poynting` for the original Standard-only full-step correlation.

Frames are empty when `GENERATE_VIDEO = False` (the default). To produce both
`TimeReversal/frames` + `tr.mp4` and `ReverseTimeMigration/frames` + `rtm.mp4`:

```bash
uv run python -m scripts.real_workflow --emitters 32 --generate-video --animation-step 1500
```

The RTM video includes the Standard/Poynting comparison. Frames and videos are
replaced for each transmitter and therefore show the last processed emitter;
the final FMC comparison sums all selected emitters. A label of `1 emitter`
means one transmitter was migrated, with every receiver used.

The immersion model follows the [example notebook](https://colab.research.google.com/drive/1vLUmfbmtpuQCr2o68tJ2GWtZq1v0cce_):
it corrects the contact label and fits the tilted water/steel interface.
It uses the first 60 µs of the 100 µs recording and mutes the strong front-wall
echo before 40 µs to emphasize the notebook's 47–72 mm hole region (set
`mute_before_us` to zero to retain shallower echoes);
the contact model uses all 30 µs and the file's 6350 m/s sound speed.
Both use a 2–6 MHz bandpass, a tapered direct-arrival mute, and an estimated
5 MHz source pulse. Real-data RTM now correlates stored forward-source pressure
with the recordings propagated backward, rather than replaying only two terminal
TR frames after absorbing boundaries have discarded wavefield information.
Correlation is sampled every 32 ns; propagation still uses every simulation time
step. The immersion propagator uses eighth-order staggered spatial derivatives
to reduce dispersion in water; contact and synthetic propagation retain second order.
The contact ROI spans 180 mm horizontally (approximately −90 to +90 mm)
and 100 mm in depth. Pressure/direction checkpoints temporarily need about
32.8 GiB RAM for immersion and 14.6 GiB for contact, released between emitters.
With `--no-poynting`, this falls to about 16.4 GiB and 7.3 GiB respectively.
Use `--output-root outputs/analysis/panther_trial` to preserve existing full-FMC
results when testing a different grid or a subset of transmitters.
These are scalar acoustic trial reconstructions: source
timing is uncalibrated, and shear waves, density contrast, and the contact
specimen's outer shape are not modelled. The immersion grid has only about six
cells per central wavelength in water; the higher-order stencil reduces spatial
dispersion but does not replace a grid/time convergence check. Full FMC runs can take much longer
than a single-transmitter trial.

Generated files are kept under `outputs/`, so the project root stays focused on code and configuration.

RTM output layout: `ReverseTimeMigration/` (real) and `SyntheticRTM/`
(synthetic) keep PNG previews and videos at the root, animation frames in
`frames/`, and all NumPy arrays in `data/` (including per-emitter results, FMC
stacks, source energy, and transmitter indices). When loading arrays from
existing runs, include `data/` in the path.

To measure the 37 saved final migration cases using thesis section 2.2.4:

```bash
uv run python -m scripts.image_quality_metrics
```

Images, review sheets, full-resolution masks, and contrast/CNR/API tables go to
`outputs/analysis/image_quality_20261005/`. Contrast and CNR are reported as
`contrast_db = 10·log10(contrast)` and `cnr_db = 20·log10(cnr)`; JSON/CSV retain
`contrast` and `cnr` as linear references. Images display signed amplitude on a
linear `seismic` scale, with symmetric limits at the reflector-window 99.5th
percentile shared across methods and boundary variants. Metrics and the API
highlight use the envelope/power data. API uses the reflector window and
retains the full-image value for reference. Signal/noise regions are proposed
in `regions.json`; edit them and rerun with `--cases 8 12` to revise those cases.
The legacy synthetic wavelength is inferred from the current source/workflow.
