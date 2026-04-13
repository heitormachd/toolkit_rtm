# Acoustic Imaging Toolkit - Documentation

GPU-accelerated Reverse Time Migration (RTM) toolkit for acoustic/seismic imaging using WebGPU compute shaders. Supports both real recorded data and fully synthetic simulations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Core Components](#3-core-components)
4. [Mathematical Foundation](#4-mathematical-foundation)
5. [Workflows](#5-workflows)
6. [Configuration & Parameters](#6-configuration--parameters)
7. [Data Formats](#7-data-formats)
8. [Dependencies & Setup](#8-dependencies--setup)
9. [Usage](#9-usage)

---

## 1. Project Overview

### What is Reverse Time Migration?

Reverse Time Migration (RTM) is a seismic/acoustic imaging technique that reconstructs subsurface reflector locations by correlating two wavefields:

- **Downgoing wavefield**: Forward-propagated from the source
- **Upgoing wavefield**: Back-propagated (time-reversed) from receiver recordings

The imaging condition crosscorrelates these two fields at every grid point and time step, producing a reflectivity image of the medium.

### What This Toolkit Does

This toolkit implements the complete RTM workflow on the GPU:

1. **Forward acoustic simulation** - Propagate a source wavelet through a velocity model, record at receivers
2. **Time reversal** - Back-propagate recorded data to reconstruct the upgoing wavefield
3. **RTM imaging** - Crosscorrelate downgoing and upgoing wavefields to image reflectors

Two imaging conditions are supported:
- **Standard RTM**: Simple pressure crosscorrelation
- **Poynting vector RTM**: Energy-flux-filtered crosscorrelation that suppresses artifacts from multiples

---

## 2. Architecture

### File Structure

```
acoustics_imaging_pv/
├── Entry Points
│   ├── main.py                            # Real/recorded data workflow
│   ├── main_synthetic.py                  # Synthetic data workflow
│   ├── main_aux.py                        # Auxiliary plotting
│   └── rtm.py                            # Alternative Taichi-based RTM
│
├── Core Framework
│   ├── SimulationConfig.py                # Base config (grid, CFL, CPML)
│   ├── WebGpuHandler.py                   # GPU pipeline management
│   └── InputTest.py                       # Data loading (MATLAB, M2K)
│
├── Simulation Classes (Real Data)
│   ├── TimeReversal.py                    # Time reversal simulation
│   └── ReverseTimeMigration.py            # RTM simulation
│
├── Simulation Classes (Synthetic Data)
│   ├── SyntheticAcouSim.py                # Forward acoustic simulation
│   ├── SyntheticTimeReversal.py           # Time reversal on synthetic data
│   └── SyntheticReverseTimeMigration.py   # RTM with Poynting vectors
│
├── GPU Shaders (WGSL)
│   ├── synthetic_acou_sim.wgsl            # Forward simulation kernels
│   ├── time_reversal.wgsl                 # Time reversal kernels
│   └── reverse_time_migration.wgsl        # RTM kernels (+ Poynting)
│
├── Utilities
│   ├── functions.py                       # Image I/O, video, plotting
│   └── signal_processing_functions.py     # FFT, filtering, windowing
│
├── Data Files
│   ├── source.npy                         # Pre-computed source waveform
│   └── map.png                            # Velocity model (color-coded)
│
├── Output Directories (generated at runtime)
│   ├── SyntheticAcouSim/
│   ├── SyntheticTR/
│   ├── SyntheticRTM/
│   ├── TimeReversal/
│   └── ReverseTimeMigration/
│
└── requirements.txt
```

### Class Hierarchy

```
SimulationConfig (base)
├── TimeReversal           ─ real data time reversal
├── ReverseTimeMigration   ─ real data RTM
├── SyntheticAcouSim       ─ synthetic forward modeling
├── SyntheticTimeReversal  ─ synthetic time reversal
└── SyntheticReverseTimeMigration  ─ synthetic RTM + Poynting
```

All simulation classes inherit from `SimulationConfig`, which initializes the grid, pressure fields, spatial derivatives, and CPML absorbing boundary parameters. Each subclass adds its own `setup_gpu()` and `run()` methods.

### Simulator Flowchart

![Simulator flowchart](flowchart_simulator.png)

> Run `generate_flowchart.py` to regenerate `flowchart_simulator.pdf` (vector) and `flowchart_simulator.png` (300 dpi raster).

**Data transferred at each CPU / GPU boundary:**

| Direction | When | Data |
|-----------|------|------|
| CPU $\rightarrow$ GPU | Initialisation | Velocity model $c(z,x)$, source / microphone signals, CPML coefficients, zero-initialised pressure fields |
| GPU $\rightarrow$ CPU | Each timestep | Pressure field $p^{n+1}$ (single readback) |
| GPU $\rightarrow$ CPU | End of simulation | Final pressure frames for downstream pipeline stages |

### Data Flow

**Synthetic Pipeline:**
```
map.png (velocity model)
    │
    ▼
SyntheticAcouSim ──► microphones_recording.npy (synthetic B-scan)
    │                         │
    │                         ▼
    │              SyntheticTimeReversal ──► last_frame.npy, second_to_last_frame.npy
    │                                                │
    │                                                ▼
    └──────────────────────────► SyntheticReverseTimeMigration
                                         │
                                         ▼
                              accumulated_product.npy (standard RTM)
                              accumulated_product_poynting.npy (Poynting RTM)
```

**Real Data Pipeline:**
```
.mat or .m2k file (recorded data)
    │
    ▼
InputTest ──► bscan (preprocessed)
    │
    ▼
TimeReversal ──► last_frame.npy, second_to_last_frame.npy
    │                      │
    │                      ▼
    └──────► ReverseTimeMigration ──► accumulated_product.npy
```

---

## 3. Core Components

### SimulationConfig.py

Base configuration class for all simulations. Initializes:

| Attribute | Description |
|-----------|-------------|
| `dt, dz, dx` | Time and spatial step sizes |
| `c` | 2D velocity model (grid_size_z x grid_size_x) |
| `grid_size_z, grid_size_x` | Grid dimensions in pixels |
| `total_time` | Number of time steps |
| `p_future, p_present, p_past` | Pressure at three time levels (leapfrog scheme) |
| `dp_1_z, dp_1_x` | 1st-order spatial derivatives (forward differences) |
| `dp_2_z, dp_2_x` | 2nd-order spatial derivatives (backward differences) |
| `absorption_layer_size` | CPML layer thickness (default: 45 pixels) |
| `psi_z, psi_x, phi_z, phi_x` | CPML auxiliary memory variables |
| `absorption_z, absorption_x` | CPML absorption coefficients |

Computes and prints the CFL stability number on initialization.

### WebGpuHandler.py

Manages the GPU compute pipeline via the `wgpu` library.

**Key methods:**

| Method | Description |
|--------|-------------|
| `__init__(shader_file)` | Load WGSL shader, auto-select workgroup sizes (1-15 per dimension) |
| `create_shader_module()` | Compile WGSL shader code |
| `create_compute_pipeline(entry_point)` | Create compute pipeline for a kernel |
| `create_buffers(data)` | Create GPU storage buffers from Python dict, parse shader bindings |
| `create_pipeline_layout()` | Build pipeline layout from bind groups |

**Helper function:**
- `read_shader_bindings(shader_lines)`: Parses `@binding` and `@group` WGSL decorators to extract buffer metadata.

### InputTest.py

Loads and preprocesses recorded acoustic data.

**Key methods:**

| Method | Description |
|--------|-------------|
| `load_data_acude(file, resampled)` | Load MATLAB `.mat` files (acude reservoir data) |
| `load_data_panther(file_m2k)` | Load `.m2k` format (Panther ultrasonic scanner) |
| `select_fmc_emitter(index)` | Extract single emitter from Full Matrix Capture data |
| `select_bscan_interval(min_t, max_t)` | Crop time axis |
| `resample_bscan(dt_new)` | Cubic interpolation to new sampling rate |
| `process_bscan()` | Preprocessing: normalize, FFT, low-pass filter at 180 Hz, IFFT |

### TimeReversal.py

Time reversal simulation for real recorded data. Flips the B-scan in time and back-propagates it through the velocity model by injecting the reversed signals at microphone positions.

**Output:** Final pressure frames for RTM input, L2-norm energy distribution.

### ReverseTimeMigration.py

RTM imaging for real data. Propagates two wavefields simultaneously:
- Downgoing: source-driven forward propagation
- Upgoing: loaded from TimeReversal output (backward in time)

Accumulates the crosscorrelation product at each time step.

**Output:** `accumulated_product.npy` — the RTM image.

### SyntheticAcouSim.py

Forward wave simulation generating synthetic receiver data. Propagates a source wavelet through a velocity model and records the pressure at microphone positions.

**Output:** `microphones_recording.npy` — synthetic B-scan.

### SyntheticTimeReversal.py

Time reversal on synthetic data. Same as `TimeReversal` but:
- Replaces reflector velocity (c=0) with medium velocity for backpropagation
- Blanks out the first 1000 samples to suppress direct arrivals

### SyntheticReverseTimeMigration.py

RTM on synthetic data with **Poynting vector imaging**. Produces two images:

1. **Standard RTM**: $I(z,x) = \sum_t p_{\downarrow}(z,x,t) \cdot p_{\uparrow}(z,x,t)$
2. **Poynting RTM**: Same product, but only where energy flux indicates a true reflection:
   - Downgoing Poynting z-component $S_z^{\downarrow} > 0$ (energy moving down)
   - Upgoing Poynting z-component $S_z^{\uparrow} < 0$ (energy moving up)

Includes velocity field computation for the Poynting vector calculation.

### functions.py

Utility functions for visualization and I/O.

| Function                        | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `convert_image_to_matrix(path)` | Parse color-coded velocity model PNG to numerical grid                        |
| `save_image(image, path)`       | Save image with even width (video codec compatibility)                        |
| `create_video(path, output)`    | Generate MP4 from frame sequence via ffmpeg (H.264, 25 fps)                   |
| `save_rtm_image(...)`           | Save 2x2 subplot (upgoing, product, downgoing, accumulated)                   |
| `plot_accumulated_product()`    | Post-processing: sum RTM images across emitters, compare standard vs Poynting |
| `plot_source()`                 | Visualize source waveform                                                     |
| `plot_l2_norm()`                | Visualize TR energy distribution                                              |

### signal_processing_functions.py

| Function | Description |
|----------|-------------|
| `low_pass_filter(signal, dt, fc)` | FFT-based low-pass filter with hard cutoff |
| `synthesize_signal(signal, peaks, ...)` | Time-align multi-receiver signals via circular shift |
| `blackman_window(signal, centers, wavelength)` | Windowed FFT around pulse centers |

### rtm.py (Alternative Implementation)

Standalone RTM implementation using the **Taichi** framework instead of WebGPU:
- Uses 8th-order finite difference stencils (higher accuracy)
- Loads Marmousi velocity model (`cp_rtm.npy`)
- Includes both standard and Poynting vector RTM
- Interactive visualization window

---

## 4. Mathematical Foundation

### 2D Acoustic Wave Equation

The toolkit solves the constant-density 2D acoustic wave equation:

$$\frac{\partial^2 p}{\partial t^2} = c(z,x)^2 \left( \frac{\partial^2 p}{\partial z^2} + \frac{\partial^2 p}{\partial x^2} \right)$$

where $p(z,x,t)$ is the pressure field and $c(z,x)$ is the spatially varying velocity.

### Finite Difference Scheme

**Time discretization** — 2nd-order leapfrog (3-level explicit):

$$p^{n+1} = 2p^n - p^{n-1} + c^2 \,\Delta t^2 \left( \frac{\partial^2 p^n}{\partial z^2} + \frac{\partial^2 p^n}{\partial x^2} \right)$$

**Spatial discretization** — 2nd-order differences computed in two passes:

**Pass 1 — Forward differences (1st derivatives):**

$$\partial_z^{(1)} p\,[z,x] = \frac{p^n[z+1,x] - p^n[z,x]}{\Delta z}, \qquad \partial_x^{(1)} p\,[z,x] = \frac{p^n[z,x+1] - p^n[z,x]}{\Delta x}$$

**Pass 2 — Backward differences (2nd derivatives):**

$$\partial_z^{(2)} p\,[z,x] = \frac{\partial_z^{(1)} p\,[z,x] - \partial_z^{(1)} p\,[z-1,x]}{\Delta z}, \qquad \partial_x^{(2)} p\,[z,x] = \frac{\partial_x^{(1)} p\,[z,x] - \partial_x^{(1)} p\,[z,x-1]}{\Delta x}$$

**Time update:**

$$p^{n+1}[z,x] = c[z,x]^2 \left(\partial_z^{(2)} p + \partial_x^{(2)} p\right) \Delta t^2 + 2\,p^n[z,x] - p^{n-1}[z,x]$$

Then shift: $p^{n-1} \leftarrow p^n$, $p^n \leftarrow p^{n+1}$.

> **Note:** The Taichi implementation in `rtm.py` uses 8th-order spatial stencils for higher accuracy.

### CFL Stability Condition

The explicit scheme is conditionally stable. The CFL number must satisfy:

$$\text{CFL} = c_{\max} \cdot \Delta t \cdot \left(\frac{1}{\Delta z} + \frac{1}{\Delta x}\right) < 1$$

This is computed and printed at initialization in `SimulationConfig.__init__()`.

### CPML Absorbing Boundary

Convolutional Perfectly Matched Layer (CPML) absorbs outgoing waves at domain boundaries, preventing non-physical reflections.

**Parameters:**
- Layer thickness: 45 pixels
- Damping coefficient: $\gamma = 3 \times 10^8$

**Absorption coefficient** — quadratic profile over the layer:

$$\alpha = \exp\!\left(-\gamma \,\xi^2 \,\Delta t\right), \qquad \xi = \frac{d}{L} \in [0,\,1]$$

where $d$ is the distance from the inner edge of the layer and $L$ is the layer thickness.

**Auxiliary field updates (applied inside the absorption layer):**

$$\phi \;\leftarrow\; \alpha\,\phi + (\alpha - 1)\,\partial^{(1)} p \qquad \text{(after forward differences)}$$
$$\partial^{(1)} p \;\mathrel{+}=\; \phi$$

$$\psi \;\leftarrow\; \alpha\,\psi + (\alpha - 1)\,\partial^{(2)} p \qquad \text{(after backward differences)}$$
$$\partial^{(2)} p \;\mathrel{+}=\; \psi$$

Four auxiliary fields ($\psi_z,\, \psi_x,\, \phi_z,\, \phi_x$) store the CPML memory variables.

### RTM Imaging Condition

**Standard RTM** — Zero-lag crosscorrelation:

$$I(z,x) = \sum_t p_{\downarrow}(z,x,t) \cdot p_{\uparrow}(z,x,t)$$

**Poynting Vector RTM** — Energy-flux-filtered crosscorrelation:

$$S_z^{\downarrow} = -p_{\downarrow}\, v_z^{\downarrow}, \qquad S_z^{\uparrow} = -p_{\uparrow}\, v_z^{\uparrow}$$

$$I_{\text{Poynting}}(z,x) = \sum_t p_{\downarrow} \cdot p_{\uparrow} \cdot \mathbb{1}\!\left[S_z^{\downarrow} > 0 \;\wedge\; S_z^{\uparrow} < 0\right]$$

The Poynting condition selects only locations where:
- Downgoing energy flows into the medium ($S_z^{\downarrow} > 0$)
- Upgoing energy flows out of the medium ($S_z^{\uparrow} < 0$)

This suppresses artifacts from multiples and backscattered noise.

**Particle velocity** is updated from the pressure gradient via:

$$v_z \;\leftarrow\; v_z - \Delta t \cdot \partial_z^{(1)} p, \qquad v_x \;\leftarrow\; v_x - \Delta t \cdot \partial_x^{(1)} p$$

---

## 5. Workflows

### Synthetic Data Workflow (`main_synthetic.py`)

```
1. Load velocity model from map.png
   └── convert_image_to_matrix() extracts velocity grid + microphone positions

2. For each microphone emitter (index 0..N):

   a. Forward Simulation (SyntheticAcouSim)
      ├── Propagate source wavelet through velocity model on GPU
      ├── Record pressure at all receiver positions at each time step
      └── Save synthetic B-scan → microphones_recording.npy

   b. Time Reversal (SyntheticTimeReversal)
      ├── Flip B-scan in time
      ├── Replace reflector velocity (c=0) with medium velocity (1500 m/s)
      ├── Suppress direct arrivals (blank first 1000 samples)
      ├── Back-propagate on GPU
      └── Save final frames → last_frame.npy, second_to_last_frame.npy

   c. RTM Imaging (SyntheticReverseTimeMigration)
      ├── Propagate downgoing wavefield (source-driven)
      ├── Load upgoing wavefield from TR frames
      ├── Accumulate standard + Poynting crosscorrelation products
      └── Save → accumulated_product_{i}.npy, accumulated_product_poynting_{i}.npy

3. Post-processing (plot_accumulated_product)
   ├── Sum accumulated products across all emitters
   ├── Display standard RTM vs Poynting RTM side-by-side
   └── Overlay ground-truth reflector positions
```

### Real Data Workflow (`main.py`)

```
1. Load recorded data
   ├── 'acude': load MATLAB .mat file via InputTest.load_data_acude()
   └── 'panther': load .m2k file via InputTest.load_data_panther()

2. Preprocess B-scan
   └── process_bscan(): normalize → FFT → low-pass filter (180 Hz) → IFFT

3. Set up uniform velocity model (c = 1500 m/s)

4. For each emitter:

   a. Time Reversal (TimeReversal)
      ├── Flip B-scan in time
      ├── Back-propagate on GPU with microphone injection
      └── Save final frames + L2-norm energy

   b. RTM Imaging (ReverseTimeMigration)
      ├── Forward-propagate source signal
      ├── Load upgoing wavefield from TR
      ├── Accumulate crosscorrelation product
      └── Save → accumulated_product_{i}.npy

5. Optionally generate videos from frame sequences
```

---

## 6. Configuration & Parameters

### Grid Parameters

| Parameter | Real Data (main.py) | Synthetic (main_synthetic.py) |
|-----------|--------------------|-----------------------------|
| `dz` | 0.01 m | $1.5 \times 10^{-3}$ m |
| `dx` | 0.01 m | $1.5 \times 10^{-3}$ m |
| `dt` | varies | varies |
| `grid_size_z` | 1000 px | from image height |
| `grid_size_x` | 4500 px | from image width |
| Depth | 10 m | from image |
| Width | 45 m | from image |

### Velocity Model

| Source | Method |
|--------|--------|
| Real data | Uniform: `c = 1500 m/s` everywhere |
| Synthetic | Color-coded PNG image (see [Data Formats](#7-data-formats)) |

### CPML Settings

| Parameter | Value |
|-----------|-------|
| `absorption_layer_size` | 45 pixels |
| `damping_coefficient` | $\gamma = 3 \times 10^8$ |
| Profile | Quadratic: $\left(\frac{d}{L}\right)^2$ |

### GPU Workgroup Sizes

Automatically selected by `WebGpuHandler` to evenly divide the grid dimensions. Each dimension tries values from 15 down to 1, selecting the largest that divides evenly.

---

## 7. Data Formats

### Input

| Format | Source | Loaded By |
|--------|--------|-----------|
| `.mat` (MATLAB) | Acude reservoir recordings | `InputTest.load_data_acude()` |
| `.m2k` | Panther ultrasonic scanner (FMC) | `InputTest.load_data_panther()` |
| `.npy` | Pre-computed source waveform | `np.load('source.npy')` |
| `.png` | Color-coded velocity model | `functions.convert_image_to_matrix()` |

### Velocity Model Color Encoding (map.png)

| Color | Velocity (m/s) | Meaning |
|-------|----------------|---------|
| White | 1500 | Receptor/microphone position |
| Blue | 1500 | Water / reference medium |
| Green | 3200 | Medium velocity material |
| Red | 6400 | High velocity material |
| Black | 0 | Reflector / boundary |

### Output

| File | Content |
|------|---------|
| `microphones_recording.npy` | Synthetic B-scan (receivers x time) |
| `last_frame.npy` | Final pressure field from time reversal |
| `second_to_last_frame.npy` | Penultimate pressure field from TR |
| `accumulated_product_{i}.npy` | Standard RTM image for emitter i |
| `accumulated_product_poynting_{i}.npy` | Poynting RTM image for emitter i |
| `l2_norm.npy` | L2-norm energy from time reversal |
| `frames/*.png` | Animation frames (sequential) |
| `*.mp4` | Simulation videos (H.264, 25 fps) |

---

## 8. Dependencies & Setup

### Python Packages

From `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `wgpu` | 0.18.1 | WebGPU compute (GPU backend) |
| `numpy` | 2.1.1 | Numerical arrays |
| `scipy` | 1.14.1 | FFT, interpolation |
| `matplotlib` | 3.9.2 | Plotting |
| `Pillow` | 10.4.0 | Image I/O |
| `ffmpeg-python` | 0.2.0 | Video generation wrapper |
| `PyQt5` | 5.15.11 | GUI backend (optional) |
| `cffi` | 1.17.1 | C FFI bindings (wgpu dependency) |

### External Tools

- **ffmpeg**: Required for video generation. Must be installed system-wide.
- **framework.file_m2k**: External module for Panther `.m2k` data (not included in repo).

### Installation

```bash
pip install -r requirements.txt
```

Ensure `ffmpeg` is available on PATH for video generation.

---

## 9. Usage

### Running the Synthetic Workflow

```bash
python main_synthetic.py
```

Requires:
- `map.png` in the project root (color-coded velocity model)
- `source.npy` in the project root (source waveform)

Produces output in `SyntheticAcouSim/`, `SyntheticTR/`, and `SyntheticRTM/` directories.

### Running the Real Data Workflow

```bash
python main.py
```

Requires:
- Recorded data file (`.mat` for acude, `.m2k` for panther)
- `source.npy` in the project root

Edit `main.py` to select dataset (`'acude'` or `'panther'`) and configure grid parameters.

### Plotting Results

```bash
python main_aux.py
```

Loads accumulated RTM products from all emitters, sums them, and displays standard vs Poynting RTM comparison with ground-truth reflector overlay.

### GPU Compute Kernels

Each simulation dispatches the following compute kernels per time step:

| Kernel | Function |
|--------|----------|
| `forward_diff` | Compute 1st-order spatial derivatives (forward differences) |
| `after_forward` | Apply CPML absorption to 1st derivatives |
| `backward_diff` | Compute 2nd-order spatial derivatives (backward differences) |
| `after_backward` | Apply CPML absorption to 2nd derivatives |
| `sim` | Update pressure via wave equation + inject source/microphone data |
| `sim_flipped_tr` | Update upgoing wavefield (RTM only) |
| `update_velocity` | Compute particle velocity from pressure (Poynting RTM only) |
| `update_rtm_image` | Accumulate Poynting-filtered product (Poynting RTM only) |
| `incr_time` | Shift time levels: past <- present <- future |
