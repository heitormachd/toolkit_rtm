"""
generate_flowchart.py
Generates a publication-quality two-column swim-lane flowchart of the
GPU acoustic simulator architecture.

Outputs
-------
flowchart_simulator.pdf   – vector (embed directly in LaTeX)
flowchart_simulator.png   – 300 dpi raster (for Word / quick preview)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import matplotlib as mpl

# ── global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Arial', 'DejaVu Serif', 'Georgia'],
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth':  0.6,
})

LC   = '#000000'   # line / border colour
TXT  = '#000000'   # text colour
F_CPU = '#f2f2f2'  # CPU box fill (light grey)
F_GPU = '#ffffff'  # GPU box fill (white)
F_TE  = '#d8d8d8'  # terminal (Start / End) fill
F_DEC = '#fafafa'  # decision diamond fill

# ── figure ────────────────────────────────────────────────────────────────────
W, H = 7.0, 11.2   # inches  (fits a two-column IEEE figure nicely)
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── layout constants ──────────────────────────────────────────────────────────
MID   = W / 2          # column separator x
C_L   = W * 0.25       # left  column centre  (CPU)
C_R   = W * 0.75       # right column centre  (GPU)
BW    = W * 0.41       # box width
BH    = 0.52           # box height
FS    = 8.5            # base font size
SMALL = FS - 1.5       # label font size

# ── swim-lane backgrounds and headers ────────────────────────────────────────
for x0, label in [(0, 'CPU  (Python)'), (MID, 'GPU  (WGSL Kernels)')]:
    # background
    ax.add_patch(mpatches.Rectangle(
        (x0 + 0.06, 0.10), MID - 0.12, H - 0.20,
        lw=0.6, edgecolor='#888888', facecolor='#fafafa', zorder=0))
    # header bar
    ax.add_patch(mpatches.Rectangle(
        (x0 + 0.06, H - 0.54), MID - 0.12, 0.44,
        lw=0.6, edgecolor='#888888', facecolor='#dddddd', zorder=1))
    ax.text(x0 + MID / 2, H - 0.32, label,
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color=TXT, zorder=2)

# dashed separator
ax.axvline(x=MID, color='#aaaaaa', lw=0.7, ls='--',
           ymin=0.10 / H, ymax=(H - 0.10) / H, zorder=1)

# ── drawing helpers ───────────────────────────────────────────────────────────

def box(cx, cy, lines, fill=F_CPU, w=BW, h=BH):
    """Rounded rectangle with multi-line text."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.05', lw=0.7,
        edgecolor=LC, facecolor=fill, zorder=3))
    ax.text(cx, cy, lines, ha='center', va='center',
            fontsize=FS, color=TXT, zorder=4,
            multialignment='center', linespacing=1.4)

def terminal(cx, cy, text):
    """Stadium / capsule shape for Start / End."""
    ax.add_patch(Ellipse(
        (cx, cy), BW * 0.68, BH * 0.80,
        lw=0.7, edgecolor=LC, facecolor=F_TE, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=FS, fontweight='bold', color=TXT, zorder=4)

def diamond(cx, cy, lines):
    """Decision diamond."""
    dw, dh = BW * 0.52, BH * 0.90
    xs = [cx, cx + dw, cx, cx - dw, cx]
    ys = [cy + dh, cy, cy - dh, cy, cy + dh]
    ax.fill(xs, ys, facecolor=F_DEC, edgecolor=LC, lw=0.7, zorder=3)
    ax.text(cx, cy, lines, ha='center', va='center',
            fontsize=FS - 0.5, color=TXT, zorder=4,
            multialignment='center', linespacing=1.4)

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=LC,
                                lw=0.85, mutation_scale=9))

def label_arrow(x, y, text, ha='left'):
    ax.text(x, y, text, ha=ha, va='center',
            fontsize=SMALL, color='#444444')

# ── y-coordinates (from top down) ────────────────────────────────────────────
GAP = 0.80   # vertical pitch between nodes

y_start  = H - 0.95
y_load   = y_start  - GAP
y_init   = y_load   - GAP
y_alloc  = y_init   - GAP
y_fwd    = y_alloc  - GAP * 0.85   # first GPU kernel, slightly closer
y_afwd   = y_fwd    - GAP
y_bwd    = y_afwd   - GAP
y_abwd   = y_bwd    - GAP
y_sim    = y_abwd   - GAP
y_incr   = y_sim    - GAP
y_dec    = y_incr   - GAP * 1.10   # decision needs a bit more room
y_copy   = y_dec                    # same row, left column
y_save   = y_copy   - GAP
y_end    = y_save   - GAP * 0.90

# ── nodes ─────────────────────────────────────────────────────────────────────
terminal(C_L, y_start, 'Start')

box(C_L, y_load,  'Load data\nSet simulation parameters')
box(C_L, y_init,  'Initialize computational grid\nCompute CPML absorption profile')
box(C_L, y_alloc, 'Allocate GPU buffers\nTransfer velocity model, source\n& sensor data to device',
    h=BH * 1.25)

box(C_R, y_fwd,  'Compute 1st-order spatial derivatives\n'
                  r'$\partial_z^{(1)}p$,  $\partial_x^{(1)}p$', fill=F_GPU, h=BH * 1.15)
box(C_R, y_afwd, 'Apply CPML correction to 1st derivatives\n'
                  r'$\phi \leftarrow \alpha\,\phi + (\alpha-1)\,\partial^{(1)}p$',
    fill=F_GPU, h=BH * 1.15)
box(C_R, y_bwd,  'Compute 2nd-order spatial derivatives\n'
                  r'$\partial_z^{(2)}p$,  $\partial_x^{(2)}p$', fill=F_GPU, h=BH * 1.15)
box(C_R, y_abwd, 'Apply CPML correction to 2nd derivatives\n'
                  r'$\psi \leftarrow \alpha\,\psi + (\alpha-1)\,\partial^{(2)}p$',
    fill=F_GPU, h=BH * 1.15)
box(C_R, y_sim,  'Wave equation update + source injection\n'
                  r'$p^{n+1} = 2p^n - p^{n-1} + c^2\Delta t^2\nabla^2 p^n$',
    fill=F_GPU, h=BH * 1.15)
box(C_R, y_incr, 'Shift pressure time levels\n'
                  r'$p^{n-1} \leftarrow p^n, \quad p^n \leftarrow p^{n+1}$',
    fill=F_GPU, h=BH * 1.15)

diamond(C_R, y_dec, 'End of\nsimulation?')

box(C_L, y_copy, 'Copy final pressure frames\nfrom GPU to host')
box(C_L, y_save, 'Save results (.npy)\nGenerate video frames')
terminal(C_L, y_end, 'End')

# ── arrows ────────────────────────────────────────────────────────────────────

# CPU top chain
arrow(C_L, y_start - BH * 0.40, C_L, y_load  + BH / 2)
arrow(C_L, y_load  - BH / 2,    C_L, y_init  + BH / 2)
arrow(C_L, y_init  - BH / 2,    C_L, y_alloc + BH * 0.625)

# Handoff CPU → GPU (L-shaped connector: down → right → up, one arrowhead at GPU entry)
h_y = (y_alloc - BH * 0.625 + y_fwd + BH * 0.575) / 2  # elbow y
ax.plot([C_L, C_L], [y_alloc - BH * 0.625, h_y], color=LC, lw=0.85)
ax.plot([C_L, C_R], [h_y, h_y],                  color=LC, lw=0.85)
arrow(C_R, h_y, C_R, y_fwd + BH * 0.575)  # single arrowhead at GPU entry

# GPU kernel chain
for (ya, yb) in [
    (y_fwd,  y_afwd),
    (y_afwd, y_bwd),
    (y_bwd,  y_abwd),
    (y_abwd, y_sim),
    (y_sim,  y_incr),
]:
    arrow(C_R, ya - BH * 0.575, C_R, yb + BH * 0.575)

# incr_time → decision
arrow(C_R, y_incr - BH * 0.575, C_R, y_dec + BH * 0.90)

# Decision: No → right-side loop back to forward_diff
loop_x = C_R + BW / 2 + 0.30
ax.plot([C_R + BW * 0.52, loop_x], [y_dec, y_dec],           color=LC, lw=0.85)
ax.plot([loop_x, loop_x],           [y_dec, y_fwd],           color=LC, lw=0.85)
ax.annotate('', xy=(C_R + BW / 2, y_fwd), xytext=(loop_x, y_fwd),
            arrowprops=dict(arrowstyle='->', color=LC, lw=0.85, mutation_scale=9))
label_arrow(loop_x + 0.05, (y_dec + y_fwd) / 2, 'No')

# Decision: Yes → CPU copy box (horizontal elbow)
arrow(C_R - BW * 0.52, y_dec, C_L + BW / 2, y_dec)
label_arrow(MID - 0.08, y_dec + 0.12, 'Yes', ha='center')

# CPU post-loop chain
arrow(C_L, y_copy - BH / 2, C_L, y_save + BH / 2)
arrow(C_L, y_save - BH / 2, C_L, y_end  + BH * 0.40)

# ── save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
for ext in ('pdf', 'png'):
    fname = f'flowchart_simulator.{ext}'
    plt.savefig(fname, bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print(f'Saved  {fname}')

plt.show()
