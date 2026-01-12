# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mol_str import parse_molecule_string, plot_molecule
from arrow3 import cones_from_vector_field

# ===================== 全局绘图风格设置 =====================
# 【修改 1】在此处统一调细边框和刻度
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # 新增以下设置以变细边框
    "axes.linewidth": 0.3,      # 边框粗细
    "xtick.major.width": 0.3,   # x轴刻度粗细
    "ytick.major.width": 0.3,   # y轴刻度粗细
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
})

# ===================== 配置区 =====================
MASK_PERCENTILE_2D = 50
FIGSIZE = (6, 4)
HEIGHT_RATIOS = [2.2, 1.0]
HSPACE = -0.2
WSPACE = 0.5

PERCENT_CLIP_2D    = (5,95)
NORM_REF_2D        = 'premask'
LENGTH_GAMMA_2D    = 1.0
USE_EQUAL_FROM_DATA = False
X_LIM_3D = (-2.9, 2.9)
Y_LIM_3D = (-2.9, 2.9)
Z_LIM_3D = (-0.7, 0.7)

XY_XLIM, XY_YLIM = (-5, 5), (-5, 5)
XZ_XLIM, XZ_ZLIM = (-5, 5), (-5, 5)
YZ_YLIM, YZ_ZLIM = (-5, 5), (-5, 5)

H_REL_MIN = 0.50
H_REL_MAX = 1.00
MASK_PERCENTILE_3D = 80

# —— 【修改 2 & 3】二维分桶绘制参数调整 ——
# 1. 密度调整
QUIVER_STEP     = 6           # 原为5，增大步长以减少格点密度 (约减少一半)

# 2. 箭头尺寸调整 (等比例增大)
# 长度缩放因子 (数值越小，箭头越长)
QUIVER_SCALE    = 1         # 原代码硬编码为2，改为1让箭头长度翻倍

# 宽度参数 (数值翻倍以变粗)
WIDTH_MIN,  WIDTH_MAX  = 0.0050, 0.0120   # 原为 0.0025, 0.0060
LWIDTH_MIN, LWIDTH_MAX = 1.2,    3.6      # 原为 0.6, 1.8

# 箭头头部参数 (适当微调以匹配新宽度)
HEADW_MIN,  HEADW_MAX  = 3.0,    5.0      # 原为 2.5, 4.0
HEADL_MIN,  HEADL_MAX  = 4.0,    7.0      # 原为 3.5, 6.0

QUIVER_COLOR    = 'blue'
N_BINS          = 4
BIN_EDGES       = None
ALPHA_MIN,  ALPHA_MAX  = 0.7,    1.0

# —— 2D 原子点样式 ——
ATOM_STYLES_2D = {
    'C': {'color': '#3a3a3a', 'size': 30},
    'H': {'color': '#cfcfcf', 'size': 10},
}
ATOM_EDGE_COLOR_2D = '#000000'
ATOM_EDGE_WIDTH_2D = 0.5
ATOM_ALPHA_2D = 0.95

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = (np.array(xs), np.array(ys), np.array(zs))

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        M = self.axes.get_proj()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        M = self.axes.get_proj()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, M)
        return np.mean(zs)

# 统一箭头样式 (注意这里也应用细边框逻辑)
axis_mutation_scale = 4
axis_kw = dict(
    arrowstyle='-|>',
    mutation_scale=axis_mutation_scale,
    lw=0.5, # 这里的3D坐标轴也保持细线
    color='0.0',
    alpha=1.0,
    shrinkA=0,
    shrinkB=0
)

def draw_momentum_axes(ax, origin=(0,0,0), length=3.5, color='black'):
    ox, oy, oz = origin
    axis_len = 6.0
    arrow_x = Arrow3D([0, axis_len], [0, 0], [0, 0], **axis_kw)
    arrow_y = Arrow3D([0, 0], [0, axis_len], [0, 0], **axis_kw)
    arrow_z = Arrow3D([0, 0], [0, 0], [0, axis_len], **axis_kw)
    ax.add_artist(arrow_x)
    ax.add_artist(arrow_y)
    ax.add_artist(arrow_z)
    ax.text(axis_len+0.5, 0, 0, "x", fontsize=9, ha="center", va="center")
    ax.text(0, axis_len+0.5, 0, "y", fontsize=9, ha="center", va="center")
    ax.text(0, 0, axis_len+0.5, "z", fontsize=9, ha="center", va="center")

def _project_atoms_to_plane(xyz_arr, plane):
    if plane == 'xy': return xyz_arr[:, 0], xyz_arr[:, 1]
    elif plane == 'xz': return xyz_arr[:, 0], xyz_arr[:, 2]
    # 【新增】支持 zx 平面 (横轴Z, 纵轴X)
    elif plane == 'zx': return xyz_arr[:, 2], xyz_arr[:, 0] 
    elif plane == 'yz': return xyz_arr[:, 1], xyz_arr[:, 2]
    else: raise ValueError("plane must be 'xy'/'xz'/'yz'/'zx'")

def draw_atoms_2d(ax, elems, xyz, plane='xy', add_legend=False):
    elems_arr = np.array([e.upper() for e in elems])
    xyz_arr = np.asarray(xyz, dtype=float)
    Xp, Yp = _project_atoms_to_plane(xyz_arr, plane)
    for sym in ('C', 'H'):
        mask = (elems_arr == sym)
        if not np.any(mask): continue
        style = ATOM_STYLES_2D.get(sym, {'color': 'k', 'size': 60})
        ax.scatter(Xp[mask], Yp[mask], s=style['size'], c=style['color'],
                   marker='o', edgecolors=ATOM_EDGE_COLOR_2D,
                   linewidths=ATOM_EDGE_WIDTH_2D, alpha=ATOM_ALPHA_2D,
                   zorder=10, label=sym)
    if add_legend: ax.legend(frameon=False, loc='upper right', fontsize=12)

def compute_vmin_vmax_from_mag(mag, percent_clip):
    vmin = np.nanpercentile(mag, percent_clip[0])
    vmax = np.nanpercentile(mag, percent_clip[1])
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-9
    return float(vmin), float(vmax)

def normalize_uv_with_bounds(U, V, vmin, vmax, hmin=0.5, hmax=1.0, gamma=1.0, eps=1e-12):
    mag  = np.sqrt(U**2 + V**2)
    nmag = (mag - vmin) / (vmax - vmin)
    nmag = np.clip(nmag, 0.0, 1.0)
    if gamma is not None and gamma != 1.0: nmag = nmag**gamma
    L    = hmin + (hmax - hmin) * nmag
    scale = L / (mag + eps)
    return U * scale, V * scale, nmag

def set_axes_equal(ax):
    xlim = np.array(ax.get_xlim3d(), dtype=float)
    ylim = np.array(ax.get_ylim3d(), dtype=float)
    zlim = np.array(ax.get_zlim3d(), dtype=float)
    rng  = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
    ctr  = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    rad  = 0.5 * rng.max()
    ax.set_xlim3d(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim3d(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim3d(ctr[2]-rad, ctr[2]+rad)
    try: ax.set_box_aspect([1,1,1])
    except Exception: pass

def set_axes_equal_from_data(ax, X, Y, Z, pad=0.05):
    mins = np.array([np.nanmin(X), np.nanmin(Y), np.nanmin(Z)], dtype=float)
    maxs = np.array([np.nanmax(X), np.nanmax(Y), np.nanmax(Z)], dtype=float)
    ctr  = (mins + maxs) / 2.0
    rad  = 0.5 * np.max(maxs - mins) * (1.0 + pad)
    ax.set_xlim3d(ctr[0]-rad, ctr[0]+rad)
    ax.set_ylim3d(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim3d(ctr[2]-rad, ctr[2]+rad)
    try: ax.set_box_aspect([1,1,1])
    except Exception: pass

def hide_3d_axes(ax):
    ax.set_axis_off()
    try:
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
    except Exception: pass

def make_bin_edges(nbins=N_BINS, edges=BIN_EDGES):
    if edges is not None: return np.asarray(edges, dtype=float)
    return np.linspace(0.0, 1.0, nbins + 1)

def lerp(a, b, t): return a + (b - a) * t

def param_per_bin(n_bins):
    ts = np.linspace(0, 1, n_bins)
    widths  = [lerp(WIDTH_MIN,  WIDTH_MAX,  t) for t in ts]
    lws     = [lerp(LWIDTH_MIN, LWIDTH_MAX, t) for t in ts]
    headw   = [lerp(HEADW_MIN,  HEADW_MAX,  t) for t in ts]
    headl   = [lerp(HEADL_MIN,  HEADL_MAX,  t) for t in ts]
    alphas  = [lerp(ALPHA_MIN,  ALPHA_MAX,  t) for t in ts]
    return widths, lws, headw, headl, alphas

# ===================== 数据生成 =====================
try:
    with open('程序-画图/FLUX_IJ.pkl','rb') as f:
        flux_ij = pickle.load(f)
except Exception:
    flux_ij = []
    dx, dy, dz = 101, 101, 61
    xs = np.linspace(-5, 5, dx)
    ys = np.linspace(-5, 5, dy)
    zs = np.linspace(-3, 3, dz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
    for _ in range(10):
        u = -y * np.exp(-(x**2 + y**2 + z**2)/10)
        v = x * np.exp(-(x**2 + y**2 + z**2)/10)
        w = z * 0.1 * np.exp(-(x**2 + y**2)/5)
        flux_ij.append([u, v, w])

dx, dy, dz = 101, 101, 61
xs = np.linspace(-5, 5, dx)
ys = np.linspace(-5, 5, dy)
zs = np.linspace(-3, 3, dz)
x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
# 数据修正
if len(flux_ij) > 4:
    flux_ij[3] = [-arr for arr in flux_ij[3]]
if len(flux_ij) > 4:
    flux_ij[4] = [-arr for arr in flux_ij[4]]

FLOW_IDXS = [3, 5, 4]
SLICERS   = [
    (slice(None, None, 6),  slice(None, None, 6),  slice(20, -20, 10)),
    (slice(None, None, 6),  slice(15, -15, 35), slice(None, None, 5)),
    (slice(15, -15, 35),    slice(None, None, 6),  slice(None, None, 5)),
]
VIEWS = [(40, 40), (40, 40), (40, 40)]

# ===================== 画布与网格 =====================
fig = plt.figure(figsize=FIGSIZE)
gs  = fig.add_gridspec(2, 3, height_ratios=HEIGHT_RATIOS, hspace=HSPACE, wspace=WSPACE)
axs3d = [fig.add_subplot(gs[0, j], projection='3d') for j in range(3)]
axs2d = [fig.add_subplot(gs[1, j]) for j in range(3)]

# ===================== 第一行：三维 =====================
plane_normal = np.array([0., 0.6, 0.0])
plane_point  = np.array([0.0, 0.0, 0.0])

def cones_on_ax(ax, X, Y, Z, U, V, W, mask_percentile=70, global_scale=4.0):
    mag = np.sqrt(U**2 + V**2 + W**2)
    mask = mag > np.percentile(mag, mask_percentile)
    X, Y, Z, U, V, W = X[mask], Y[mask], Z[mask], U[mask], V[mask], W[mask]
    cones_from_vector_field(
        ax, X, Y, Z, U, V, W,
        aspect=13, segments=7, color='C0', alpha=0.95,
        cap=False, color_by_magnitude=False, cmap='viridis',
        global_scale=global_scale,
        h_rel_min=H_REL_MIN, h_rel_max=H_REL_MAX,
        plane_normal=plane_normal, plane_point=plane_point, draw_plane=False,
        distance_fade_strength=0.0, alpha_far_factor=1.0, distance_gamma=1.0,
        plane_size_mode='data', plane_scale=0.3, plane_pad=0.15
    )
    return X, Y, Z

benzene = (
    "C 0 1.3964 0; C 1.2093178738 0.6982 0; C 1.2093178738 -0.6982 0;"
    "C 0 -1.3964 0; C -1.2093178738 -0.6982 0; C -1.2093178738 0.6982 0;"
    "H 0 2.4719 0; H 2.1407281956 1.23595 0; H 2.1407281956 -1.23595 0;"
    "H 0 -2.4719 0; H -2.1407281956 -1.23595 0; H -2.1407281956 1.23595 0;"
)
elems, xyz = parse_molecule_string(benzene)

for col in range(3):
    ax = axs3d[col]
    hide_3d_axes(ax)
    ax.set_proj_type('ortho')
    ax.view_init(*VIEWS[col])

    sx, sy, sz = SLICERS[col]
    Xs, Ys, Zs = x[sx, sy, sz], y[sx, sy, sz], z[sx, sy, sz]
    if FLOW_IDXS[col] < len(flux_ij):
        U3 = flux_ij[FLOW_IDXS[col]][0][sx, sy, sz]
        V3 = flux_ij[FLOW_IDXS[col]][1][sx, sy, sz]
        W3 = flux_ij[FLOW_IDXS[col]][2][sx, sy, sz]
    else:
        U3, V3, W3 = np.zeros_like(Xs), np.zeros_like(Xs), np.zeros_like(Xs)

    Xc, Yc, Zc = cones_on_ax(ax, Xs, Ys, Zs, U3, V3, W3,
                             mask_percentile=MASK_PERCENTILE_3D,
                             global_scale=2.5)
    plot_molecule(elems, xyz, fig, ax)
    draw_momentum_axes(ax, origin=(0,0,0), length=3.5, color='black')

    if USE_EQUAL_FROM_DATA:
        set_axes_equal_from_data(ax, Xc, Yc, Zc, pad=0.05)
    else:
        ax.set_xlim(X_LIM_3D); ax.set_ylim(Y_LIM_3D); ax.set_zlim(Z_LIM_3D)
        set_axes_equal(ax)

axis_kw2 = dict(
    arrowstyle='-|>',
    mutation_scale=2*axis_mutation_scale,
    lw=2,
    alpha=1.0,
    color='tab:red',
    shrinkA=0,
    shrinkB=0
)

mu_len = 3.5
mu_x = Arrow3D([0, mu_len], [0, 0], [0, 0],  **axis_kw2)
mu_y = Arrow3D([0, 0], [0, mu_len], [0, 0],  **axis_kw2)
mu_z = Arrow3D([0, 0], [0, 0], [0, mu_len],  **axis_kw2)

axs3d[2].add_artist(mu_x)
axs3d[1].add_artist(mu_y)
axs3d[0].add_artist(mu_z)

axs3d[2].text(mu_len+1, 2.5, 0.0, r"$L_{x}^{(YZ)}$", fontsize=12, ha="left", va="center")
axs3d[1].text(3, mu_len+3, 0, r"$L_{y}^{(ZX)}$", fontsize=12, ha="center", va="bottom")
axs3d[0].text(+0.1, 3, mu_len+1, r"$L_{z}^{(XY)}$", fontsize=12, ha="center", va="bottom")

# ===================== 第二行：二维 =====================
def slice_field_for_quiver(component_idx_pair, plane='xy', value=0.0, step=5, case_idx=5):
    if case_idx >= len(flux_ij): return np.array([]),np.array([]),np.array([]),np.array([])
    U3, V3, W3 = (flux_ij[case_idx][0], flux_ij[case_idx][1], flux_ij[case_idx][2])
    
    # 提取切面数据
    if plane == 'xy':
        k = np.argmin(np.abs(zs - value))
        X2D, Y2D = x[:, :, k], y[:, :, k]
        vecs = {'U': U3[:, :, k], 'V': V3[:, :, k], 'W': W3[:, :, k]}
    elif plane == 'xz':
        j = np.argmin(np.abs(ys - value))
        X2D, Y2D = x[:, j, :], z[:, j, :]
        vecs = {'U': U3[:, j, :], 'V': V3[:, j, :], 'W': W3[:, j, :]}
    # 【新增】zx 平面：取 y=0 切片，但交换横纵坐标 (Z为横, X为纵)
    elif plane == 'zx':
        j = np.argmin(np.abs(ys - value))
        X2D, Y2D = z[:, j, :], x[:, j, :]
        vecs = {'U': U3[:, j, :], 'V': V3[:, j, :], 'W': W3[:, j, :]}
    elif plane == 'yz':
        i = np.argmin(np.abs(xs - value))
        X2D, Y2D = y[i, :, :], z[i, :, :]
        vecs = {'U': U3[i, :, :], 'V': V3[i, :, :], 'W': W3[i, :, :]}
    else:
        raise ValueError("plane must be 'xy'/'xz'/'yz'/'zx'")

    C1, C2 = vecs[component_idx_pair[0]], vecs[component_idx_pair[1]]
    
    return X2D[::step, ::step], Y2D[::step, ::step], C1[::step, ::step], C2[::step, ::step]

def quiver_bucketed(ax, X, Y, U, V, percent_clip=PERCENT_CLIP_2D,
                    hmin=H_REL_MIN, hmax=H_REL_MAX, color=QUIVER_COLOR,
                    bin_edges=BIN_EDGES, n_bins=N_BINS,
                    mask_percentile=MASK_PERCENTILE_2D, norm_ref=NORM_REF_2D,
                    length_gamma=LENGTH_GAMMA_2D):
    if X.size == 0: return
    mag_all = np.sqrt(U**2 + V**2)
    vmin_ref, vmax_ref = compute_vmin_vmax_from_mag(mag_all, percent_clip)

    if mask_percentile is not None and float(mask_percentile) > 0:
        thr = np.nanpercentile(mag_all, float(mask_percentile))
        keep = mag_all >= thr
        X, Y, U, V = X[keep], Y[keep], U[keep], V[keep]
        if U.size == 0: return
        if norm_ref == 'postmask':
            vmin_ref, vmax_ref = compute_vmin_vmax_from_mag(np.sqrt(U**2 + V**2), percent_clip)

    U_s, V_s, nmag = normalize_uv_with_bounds(U, V, vmin=vmin_ref, vmax=vmax_ref,
                                              hmin=hmin, hmax=hmax, gamma=length_gamma)

    edges = make_bin_edges(nbins=n_bins, edges=bin_edges)
    n_bins_eff = len(edges) - 1
    widths, lws, headw, headl, alphas = param_per_bin(n_bins_eff)

    for bi in range(n_bins_eff):
        lo, hi = edges[bi], edges[bi+1]
        mask = (nmag >= lo) & (nmag <= hi if bi == n_bins_eff-1 else nmag < hi)
        if not np.any(mask): continue
        # 【修改 3】使用 global QUIVER_SCALE 替代原有的 scale=2
        # width 和 headwidth 已经通过 param_per_bin 在上方配置区被增大了
        ax.quiver(X[mask], Y[mask], U_s[mask], V_s[mask], color=color,
                  angles='xy', scale_units='xy', scale=QUIVER_SCALE,
                  width=float(widths[bi]), linewidths=float(lws[bi]), pivot='mid',
                  headwidth=float(headw[bi]), headlength=float(headl[bi]),
                  alpha=float(alphas[bi]))

# 按照您的描述：第一列xy, 第二列xz(zx), 第三列yz
panels = [
    {"title": "xy (z=0)", "plane": "xy", "pair": ("U","V"), "case": FLOW_IDXS[0], "xlim": XY_XLIM, "ylim": XY_YLIM},
    
    # 【修改】这里改为 zx (y=0)，pair 改为 (W, U)，坐标范围互换
    {"title": "zx (y=0)", "plane": "zx", "pair": ("W","U"), "case": FLOW_IDXS[1], "xlim": XZ_ZLIM, "ylim": XZ_XLIM},
    
    {"title": "yz (x=0)", "plane": "yz", "pair": ("V","W"), "case": FLOW_IDXS[2], "xlim": YZ_YLIM, "ylim": YZ_ZLIM},
]

for ax2d, p in zip(axs2d, panels):
    X2D, Y2D, U2D, V2D = slice_field_for_quiver(p["pair"], plane=p["plane"], value=0.0, step=QUIVER_STEP, case_idx=p["case"])
    quiver_bucketed(ax2d, X2D, Y2D, U2D, V2D, percent_clip=PERCENT_CLIP_2D,
                    hmin=H_REL_MIN, hmax=H_REL_MAX, color=QUIVER_COLOR,
                    bin_edges=BIN_EDGES, n_bins=N_BINS)
    ax2d.set_title(p["title"])
    if p["xlim"] is not None: ax2d.set_xlim(p["xlim"])
    if p["ylim"] is not None: ax2d.set_ylim(p["ylim"])
    draw_atoms_2d(ax2d, elems, xyz, plane=p["plane"], add_legend=False)
    
    pad_val = -2  # 您可以尝试 -2, -3, -5 直到满意为止
    
    if p["plane"] == "xy": 
        ax2d.set_xlabel(r'$x$ (${\rm \AA}$)', labelpad=pad_val)
        ax2d.set_ylabel(r'$y$ (${\rm \AA}$)', labelpad=pad_val)
    elif p["plane"] == "xz": 
        ax2d.set_xlabel(r'$x$ (${\rm \AA}$)', labelpad=pad_val)
        ax2d.set_ylabel(r'$z$ (${\rm \AA}$)', labelpad=pad_val)
    elif p["plane"] == "zx":
        ax2d.set_xlabel(r'$z$ (${\rm \AA}$)', labelpad=pad_val)
        ax2d.set_ylabel(r'$x$ (${\rm \AA}$)', labelpad=pad_val)
    else: 
        ax2d.set_xlabel(r'$y$ (${\rm \AA}$)', labelpad=pad_val)
        ax2d.set_ylabel(r'$z$ (${\rm \AA}$)', labelpad=pad_val)

# 确保所有文字在最上层
for ax in axs3d + axs2d:
    for txt in ax.texts:
        txt.set_zorder(1e4)

plt.show()