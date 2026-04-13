import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import matplotlib as mpl

# ── estilo global ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Arial', 'DejaVu Serif', 'Georgia'],
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth':  0.6,
})

LC   = '#000000'   # cor da linha / borda
TXT  = '#000000'   # cor do texto
F_CPU = '#f2f2f2'  # preenchimento CPU (cinza claro)
F_GPU = '#ffffff'  # preenchimento GPU (branco)
F_TE  = '#d8d8d8'  # preenchimento terminal (Início / Fim)
F_DEC = '#fafafa'  # preenchimento diamante de decisão

# ── figura ────────────────────────────────────────────────────────────────────
W, H = 7.0, 11.2   # polegadas
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── constantes de layout ──────────────────────────────────────────────────────
MID   = W / 2          # column separator x
C_L   = W * 0.25       # left  column centre  (CPU)
C_R   = W * 0.75       # right column centre  (GPU)
BW    = W * 0.41       # box width
BH    = 0.52           # box height
FS    = 8.5            # base font size
SMALL = FS - 1.5       # label font size


# ── swim-lanes (raias) e cabeçalhos ──────────────────────────────────────────
for x0, label in [(0, 'CPU (Python)'), (MID, 'GPU (Kernels WGSL)')]:
    # fundo
    ax.add_patch(mpatches.Rectangle(
        (x0 + 0.06, 0.10), MID - 0.12, H - 0.20,
        lw=0.6, edgecolor='#888888', facecolor='#fafafa', zorder=0))
    # barra de cabeçalho
    ax.add_patch(mpatches.Rectangle(
        (x0 + 0.06, H - 0.54), MID - 0.12, 0.44,
        lw=0.6, edgecolor='#888888', facecolor='#dddddd', zorder=1))
    ax.text(x0 + MID / 2, H - 0.32, label,
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color=TXT, zorder=2)

# separador tracejado
ax.axvline(x=MID, color='#aaaaaa', lw=0.7, ls='--',
           ymin=0.10 / H, ymax=(H - 0.10) / H, zorder=1)

# ── funções auxiliares de desenho ─────────────────────────────────────────────

def box(cx, cy, lines, fill=F_CPU, w=BW, h=BH):
    """Retângulo arredondado com texto multi-linha."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.05', lw=0.7,
        edgecolor=LC, facecolor=fill, zorder=3))
    ax.text(cx, cy, lines, ha='center', va='center',
            fontsize=FS, color=TXT, zorder=4,
            multialignment='center', linespacing=1.4)

def terminal(cx, cy, text):
    """Forma de cápsula para Início / Fim."""
    ax.add_patch(Ellipse(
        (cx, cy), BW * 0.68, BH * 0.80,
        lw=0.7, edgecolor=LC, facecolor=F_TE, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=FS, fontweight='bold', color=TXT, zorder=4)

def diamond(cx, cy, lines):
    """Diamante de decisão."""
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

# ── coordenadas y (de cima para baixo) ───────────────────────────────────────
GAP = 0.80

y_start  = H - 0.95
y_load   = y_start  - GAP
y_init   = y_load   - GAP
y_alloc  = y_init   - GAP
y_fwd    = y_alloc  - GAP * 0.85
y_afwd   = y_fwd    - GAP
y_bwd    = y_afwd   - GAP
y_abwd   = y_bwd    - GAP
y_sim    = y_abwd   - GAP
y_incr   = y_sim    - GAP
y_dec    = y_incr   - GAP * 1.10
y_copy   = y_dec
y_save   = y_copy   - GAP
y_end    = y_save   - GAP * 0.90

# ── nós (traduções aplicadas aqui) ───────────────────────────────────────────
terminal(C_L, y_start, 'Início')

box(C_L, y_load,  'Carregar dados\nDefinir parâmetros de simulação')
box(C_L, y_init,  'Inicializar grade computacional\nCalcular perfil de absorção CPML')
box(C_L, y_alloc, 'Alocar buffers na GPU\nTransferir modelo de velocidade,\nfonte e sensores para o dispositivo',
    h=BH * 1.25)

box(C_R, y_fwd,  'Calcular derivadas espaciais de 1ª ordem\n'
                  r'$\partial_z^{(1)}p$,  $\partial_x^{(1)}p$', fill=F_GPU, h=BH * 1.15)
box(C_R, y_afwd, 'Aplicar correção CPML às derivadas de 1ª ordem\n'
                  r'$\phi \leftarrow \alpha\,\phi + (\alpha-1)\,\partial^{(1)}p$',
    fill=F_GPU, h=BH * 1.15)
box(C_R, y_bwd,  'Calcular derivadas espaciais de 2ª ordem\n'
                  r'$\partial_z^{(2)}p$,  $\partial_x^{(2)}p$', fill=F_GPU, h=BH * 1.15)
box(C_R, y_abwd, 'Aplicar correção CPML às derivadas de 2ª ordem\n'
                  r'$\psi \leftarrow \alpha\,\psi + (\alpha-1)\,\partial^{(2)}p$',
    fill=F_GPU, h=BH * 1.15)
box(C_R, y_sim,  'Atualização da eq. de onda + injeção de fonte\n'
                  r'$p^{n+1} = 2p^n - p^{n-1} + c^2\Delta t^2\nabla^2 p^n$',
    fill=F_GPU, h=BH * 1.15)
box(C_R, y_incr, 'Avançar níveis de tempo da pressão\n'
                  r'$p^{n-1} \leftarrow p^n, \quad p^n \leftarrow p^{n+1}$',
    fill=F_GPU, h=BH * 1.15)

diamond(C_R, y_dec, 'Fim da\nsimulação?')

box(C_L, y_copy, 'Copiar frames de pressão finais\nda GPU para o host (CPU)')
box(C_L, y_save, 'Salvar resultados (.npy)\nGerar frames de vídeo')
terminal(C_L, y_end, 'Fim')

# ── setas e rótulos de fluxo ─────────────────────────────────────────────────
arrow(C_L, y_start - BH * 0.40, C_L, y_load  + BH / 2)
arrow(C_L, y_load  - BH / 2,    C_L, y_init  + BH / 2)
arrow(C_L, y_init  - BH / 2,    C_L, y_alloc + BH * 0.625)

h_y = (y_alloc - BH * 0.625 + y_fwd + BH * 0.575) / 2
ax.plot([C_L, C_L], [y_alloc - BH * 0.625, h_y], color=LC, lw=0.85)
ax.plot([C_L, C_R], [h_y, h_y],                  color=LC, lw=0.85)
arrow(C_R, h_y, C_R, y_fwd + BH * 0.575)

for (ya, yb) in [(y_fwd, y_afwd), (y_afwd, y_bwd), (y_bwd, y_abwd), (y_abwd, y_sim), (y_sim, y_incr)]:
    arrow(C_R, ya - BH * 0.575, C_R, yb + BH * 0.575)

arrow(C_R, y_incr - BH * 0.575, C_R, y_dec + BH * 0.90)

# Decisão: Não (Loop)
loop_x = C_R + BW / 2 + 0.30
ax.plot([C_R + BW * 0.52, loop_x], [y_dec, y_dec],           color=LC, lw=0.85)
ax.plot([loop_x, loop_x],           [y_dec, y_fwd],           color=LC, lw=0.85)
ax.annotate('', xy=(C_R + BW / 2, y_fwd), xytext=(loop_x, y_fwd),
            arrowprops=dict(arrowstyle='->', color=LC, lw=0.85, mutation_scale=9))
label_arrow(loop_x + 0.05, (y_dec + y_fwd) / 2, 'Não')

# Decisão: Sim
arrow(C_R - BW * 0.52, y_dec, C_L + BW / 2, y_dec)
label_arrow(MID - 0.08, y_dec + 0.12, 'Sim', ha='center')

arrow(C_L, y_copy - BH / 2, C_L, y_save + BH / 2)
arrow(C_L, y_save - BH / 2, C_L, y_end  + BH * 0.40)

# ── salvar ────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
for ext in ('pdf', 'png'):
    fname = f'flowchart_simulator_ptbr.{ext}'
    plt.savefig(fname, bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    print(f'Salvo  {fname}')

plt.show()