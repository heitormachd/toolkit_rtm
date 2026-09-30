# TD-RTM paper simulation maps

Generate the five raw toolkit maps with:

```bash
uv run python scripts/generate_td_rtm_models.py
```

The outputs are `assets/models/td_rtm_t_b1.png` and
`assets/models/td_rtm_y_b1.png` through `td_rtm_y_b4.png`. Each is an RGB PNG
of shape `[z, x] = [1041, 801]`, with 0.05 mm spacing, x = −20…20 mm, z =
0…52 mm, and z positive downward. Pixel indices therefore correspond to
`x_mm = -20 + 0.05*x_index` and `z_mm = 0.05*z_index`.

The binary palette follows the toolkit: blue `#0000FF` is water and crack fill
(1500 m/s in the current parser), red `#FF0000` is aluminium (6400 m/s), and
white `#FFFFFF` is reserved for the 64 colocated TX/RX points. Markers are
single pixels at row 1 (`z = 0.05 mm`) and x indices 85, 95, …, 715 (0.5 mm
pitch). They approximate the paper's finite 0.4 mm elements while keeping the
maps compatible with the loader's horizontal colocated-array support. The paper's physical
values are 1496 m/s for water and 6320 m/s for aluminium; the rounded parser
palette is recorded here so the difference is explicit.

The interface uses the paper's Eq. (8) in plotted coordinates,
`z_mm = 14 + B_mm*sin(0.05*pi*x_mm)`, with B = 1 for T and Y-B1, then 2, 3,
and 4 mm for Y-B2 through Y-B4. The crack geometry is inferred from the
schematics: a 4 mm × 0.4 mm upward stem at an approximately 33 mm junction,
with either a horizontal T branch or two 120° Y arms. The schematics look about
1 mm deeper than the explicit equation, and the RTM result panels show the
response near 32 mm; the generated input follows Eq. (8) and the 33 mm
schematic estimate. Crack fill is blue because the paper does not specify a
crack medium.

This scope covers the simulation geometries in Figs. 4 and 6. The experimental
samples in Fig. 10 are omitted because complete machine-readable contours and
crack coordinates are not published. The paper reports k-Wave FMC forward
data followed by tenth-order spatial, second-order temporal RTM with a
20-cell PML; its simulations show increased branched-crack completeness from
multipath data, loss of the Y stem at B = 3, and failure at B = 4.

The PNGs are direct raw RGB maps: there are no axes, borders, plot previews,
or antialiasing.
