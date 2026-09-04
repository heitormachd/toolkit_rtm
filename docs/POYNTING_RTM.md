# Corrected Poynting-vector RTM

The source and receiver particle velocities remain on the solver's staggered
grid during propagation. Immediately before the imaging condition, each
component is linearly interpolated to the pressure node: the vertical
component is averaged with the sample at `z - 1`, and the horizontal component
with the sample at `x - 1`. Poynting accumulation is skipped on the first row
and column, which are part of the outer CPML boundary.

`update_velocity` advances velocity from the pressure gradients at time `n`.
The simulation kernels then retain pressure at `n` in `p_past` and pressure at
`n + 1` in `p_present`. Their average is therefore used with velocity at
`n + 1/2` for both wavefields.

The shader forms the two-dimensional vectors

```text
Js = (ps * vs_x, ps * vs_z)
Jr = (-pr * vr_x, -pr * vr_z)
```

The repository's integrated acoustic particle velocity makes `p * v` point
along the outgoing wavefront in a homogeneous-medium check; the references'
`-p * v` expression uses the opposite velocity/stress sign convention. The
source field uses the repository convention directly. The receiver field is
replayed forward from the final time-reversal state during the second RTM
pass, so its flux is negated to express the receiver propagation direction
used by the Yoon--Marfurt opening-angle condition.
The shader clamps their normalized dot product to `[-1, 1]` and accumulates
`ps * pr` exactly when `cos(theta) >= -0.5`, including the 120-degree boundary.
Each vector is scaled by its largest absolute component before normalization,
which avoids overflow/underflow without imposing an amplitude cutoff. True
zero-norm vectors do not contribute.

The legacy conventional and Poynting accumulations are still saved. Two new
outputs append `_normalized` to those names. Both divide by source illumination
accumulated from the matching half-step source pressure. The normalization
denominator is floored at 0.1% of the maximum source illumination so weakly
illuminated regions remain visible without receiving unbounded amplification.
The cropped illumination map is saved as
`accumulated_source_energy_<emitter>.npy`.

The direct-arrival mute remains 1,000 samples long, but its last 100 samples
now use a raised-cosine transition instead of an abrupt step. Standard and
Poynting panels in RTM videos share one 99.5th-percentile color limit so their
displayed amplitudes are directly comparable.

Run the angle-rule checks with:

```bash
uv run python -m unittest tests.test_poynting_rtm -v
```

Run the existing forward, time-reversal, and RTM comparison with:

```bash
uv run python -m scripts.synthetic_workflow
```

By default this performs sparse Full Matrix Capture: every 16th receiver emits
once and all receivers record the shot. Raw standard and Poynting imaging
numerators, and source illumination, are summed over shots before a single
normalization. Use `FMC_TRANSMITTER_STRIDE = 1` for full FMC or set
`ENABLE_SPARSE_FMC = False` for the bitmap-source single-shot workflow.

Set `ENABLE_POYNTING_VECTORS` near the top of `scripts/synthetic_workflow.py`
to `False` to run conventional RTM without the particle-velocity and Poynting
kernels. Conventional raw and normalized outputs are still written; existing
Poynting output files are left untouched.

For a homogeneous direction check, pass one or two sample indices through
`poynting_validation_steps` when calling `run`. The selected half-step pressure
and collocated source/receiver `Jx` and `Jz` arrays are then available in
`rtm.poynting_validation_snapshots` for pressure overlays or sparse quiver
plots. No extra readback occurs when the argument is omitted.

## Validation performed

- Direct WebGPU kernel cases retained vectors at 0, 90, and 120 degrees and
  rejected vectors at 150 and 180 degrees. A half-step pressure pair of 1 and
  3 accumulated the expected product of 4.
- Full-model samples around the reflector had source/replayed-receiver median
  cosines between -0.996 and -1.000 before correcting the replay convention.
  Negating the receiver flux retained all sampled high-energy reflection
  vectors instead of rejecting all of them.
- In a 140 x 140 homogeneous wavefront check, the collocated source Poynting
  vectors had median radial cosines of 1.0, 1.0, and 0.99994 at three selected
  steps; every sampled high-energy vector pointed outward.
- In a small horizontal-reflector model, raw signal-to-artifact ratio increased
  from 0.1438 to 0.1743 (21.2%) and normalized ratio from 0.2556 to 0.2978
  (16.5%). The hard gate retained 51.1% of raw reflector-mask energy.
- The corrected full `map.png` control used 2,800 aligned samples, a zero-mean
  Ricker source, and a 1,000-sample direct-arrival mute. Normalized Poynting
  reflector/background contrast was 58.85, and reflector energy was 2.75
  times the remaining surface artifact. All four saved images were finite.
