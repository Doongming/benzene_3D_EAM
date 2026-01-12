# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mol_str import parse_molecule_string, plot_molecule

# ------------------------ Arrow3D 定义 ------------------------
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
        # 把这个 artist 的“深度”设成一个很大的常数，让它总在最前面
        return np.max(zs) + 1e3

# 统一箭头样式
axis_mutation_scale = 4
axis_kw = dict(
    arrowstyle='-|>',
    mutation_scale=axis_mutation_scale,
    lw=0.5,
    color='0.0',
    alpha=1.0,
    shrinkA=0,
    shrinkB=0
)

# ------------------------ 数据（Gaussian D2h 标签） ------------------------
states = [
    (1,  "B3u", 5.2025, 238.31, 0.0000),
    (2,  "B2u", 6.4992, 190.77, 0.0000),
    (3,  "B2g", 6.5771, 188.51, 0.0000),
    (4,  "B3g", 6.5771, 188.51, 0.0000),
    (5,  "B1u", 7.1175, 174.20, 0.0678),
    (6,  "Au",  7.1971, 172.27, 0.0000),
    (7,  "B1u", 7.1971, 172.27, -0.0000),
    (8,  "Au",  7.2974, 169.90, 0.0000),
    (9,  "B3u", 7.3448, 168.81, 0.6806),
    (10, "B2u", 7.3448, 168.81, 0.6806),
    (11, "B2g", 7.8104, 158.74, 0.0000),
]

# ------------------------ D2h -> D6h 映射 ------------------------
d2h_to_d6h_1D = {
    "Ag":  r"A$_{1g}$", "B1g": r"A$_{2g}$", "B2g": r"B$_{1g}$", "B3g": r"B$_{2g}$",
    "Au":  r"A$_{1u}$", "B1u": r"A$_{2u}$ (Z)", "B2u": r"B$_{1u}$", "B3u": r"B$_{2u}$",
}

pair_to_E = {
    frozenset(["B2g","B3g"]): r"E$_{1g}$",
    frozenset(["Ag","B1g"]):  r"E$_{2g}$",
    frozenset(["B2u","B3u"]): r"E$_{1u}$ (X,Y)",
    frozenset(["Au","B1u"]):  r"E$_{2u}$",
}

# ------------------------ 聚类判断简并 ------------------------
tol = 1e-4
states_sorted = sorted(states, key=lambda x: x[2])
groups = []
i = 0
while i < len(states_sorted):
    idx, ir, e, w, f = states_sorted[i]
    gi, gidx = [ir], [idx]
    j = i + 1
    while j < len(states_sorted) and abs(states_sorted[j][2] - e) < tol:
        gi.append(states_sorted[j][1])
        gidx.append(states_sorted[j][0])
        j += 1
    groups.append({"energy": e, "irreps_d2h": gi, "indices": gidx})
    i = j

for g in groups:
    S = set(g["irreps_d2h"])
    if len(S) == 2 and frozenset(S) in pair_to_E:
        g["irrep_d6h"], g["degeneracy"] = pair_to_E[frozenset(S)], 2
    else:
        mapped = [d2h_to_d6h_1D.get(ir, ir) for ir in g["irreps_d2h"]]
        g["irrep_d6h"] = ",".join(sorted(set(mapped)))
        g["degeneracy"] = len(g["irreps_d2h"])

# ------------------------ 图：子图 1（能级+高斯），子图 2（分子+µ） ------------------------
fig = plt.figure(figsize=(10, 4.5))
# 宽度比例 3:2
gs  = fig.add_gridspec(1, 2, width_ratios=[2, 3], wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])                   # (a) 能级 + 高斯
ax2 = fig.add_subplot(gs[0, 1], projection='3d')  # (b) 分子 + µ

# ==================== 子图 1：能级 + 纵向高斯包络 ====================
ax1.set_axis_off()

ymin, ymax = 5.1, 8.2
arrow_x = 0.08

# 纵轴
ax1.annotate("", xy=(arrow_x, ymax), xytext=(arrow_x, ymin),
             arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"))

# 断裂双斜线
slash_len = 0.04
slash_gap = 0.05
y_center  = ymin
for offset in (-slash_gap/2, slash_gap/2):
    x0 = arrow_x - slash_len/2
    x1 = arrow_x + slash_len/2
    y0 = y_center + offset - slash_len/2
    y1 = y_center + offset + slash_len/2
    ax1.plot([x0, x1], [y0, y1], color="black", lw=1.8, solid_capstyle="butt")

# 纵轴单位
ax1.text(arrow_x + 0.02, ymax - 0.05, "E / eV",
         va="top", ha="left", fontsize=12, color="black")

# 刻度
for t in [6.0, 6.5, 7.0, 7.5, 8.0]:
    ax1.plot([arrow_x - 0.012, arrow_x + 0.012], [t, t],
             lw=1.4, color="black", solid_capstyle="butt")
    ax1.text(arrow_x - 0.035, t, f"{t:.1f}",
             va="center", ha="right", fontsize=12, color="black")

# ---- 高斯包络（只画峰值前后 1.5 eV） ----
E0   = 7.23       # 峰值中心
FWHM = 0.2        # 展宽
sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

E_min = max(ymin-1.0, E0 - 0.5)
E_max = min(ymax,     E0 + 0.5)
E_grid = np.linspace(E_min, E_max, 400)

G_raw = np.exp(-(E_grid - E0)**2 / (2.0 * sigma**2))  # 原始高斯
G = G_raw / 3.0  # 峰值降低 3 倍，用于水平延展

center_x = 0.58        # 能级中心
x_max   = 0.20         # 包络向右最大水平宽度
X_gauss = center_x + x_max * G   # 基线从能级中心移过去


# ---- 能级图 ----
full_len = 0.80
y_ground = ymin - 0.3 + 0.018

# 基态
ax1.plot([center_x - full_len/2, center_x + full_len/2], [y_ground, y_ground],
         lw=2.2, color="black", solid_capstyle="butt", clip_on=False,zorder=1)
ax1.text(center_x + full_len/2 + 0.02, y_ground,
         r"A$_{1g}$ (g)", va="center", ha="left", fontsize=12, color="black")

exc_len = full_len / 3.0

def level_color(indices):
    # 5, 9, 10 有振子强度，用黑色强调，其余淡灰
    return "black" if any(i in {5, 9, 10} for i in indices) else "#e8e8e8"

text_dx = 0.02
for g in groups:
    E   = g["energy"]
    col = level_color(g["indices"])
    if g["degeneracy"] == 2:
        gap = 0.05
        L_l, L_r = center_x - gap/2 - exc_len, center_x - gap/2
        R_l, R_r = center_x + gap/2, center_x + gap/2 + exc_len
        ax1.plot([L_l, L_r], [E, E], lw=2.0, color=col, solid_capstyle="butt",zorder=1)
        ax1.plot([R_l, R_r], [E, E], lw=2.0, color=col, solid_capstyle="butt",zorder=1)
        

        ax1.text(R_r + text_dx, E, g["irrep_d6h"],
                 va="center", ha="left", fontsize=12, color="black")           
    else:
        x_l = center_x - exc_len/2
        x_r = center_x + exc_len/2
        ax1.plot([x_l, x_r], [E, E], lw=2.0, color=col, solid_capstyle="butt",zorder=1)
        if g["energy"] -7.1175 < 0.01:
            ax1.text(x_r + text_dx, E-0.1, g["irrep_d6h"],
                 va="center", ha="left", fontsize=12, color="black")
        else:
            ax1.text(x_r + text_dx, E, g["irrep_d6h"],
                 va="center", ha="left", fontsize=12, color="black")

    
ax1.set_xlim(0, 1)
ax1.set_ylim(ymin-1, ymax)
ax1.text(0.0, ymax+0.3, "(a)", ha="left", va="top", fontsize=12)

# 包络线
ax1.plot(X_gauss, E_grid, color="blue", lw=1.4,zorder=4)

# 单一颜色填充（从能级中心到高斯曲线）
ax1.fill_betweenx(
    E_grid,
    center_x,    # 基线：能级中心
    X_gauss,     # 高斯曲线
    color="blue",
    alpha=0.3 ,   # 你可以根据视觉效果调 0.2~0.5
    zorder=4
)

# ==================== 子图 2：分子 + 3D 坐标 + µ 向量 ====================
benzene = (
    "C 0 1.3964 0; C 1.2093178738 0.6982 0; C 1.2093178738 -0.6982 0;"
    "C 0 -1.3964 0; C -1.2093178738 -0.6982 0; C -1.2093178738 0.6982 0;"
    "H 0 2.4719 0; H 2.1407281956 1.23595 0; H 2.1407281956 -1.23595 0;"
    "H 0 -2.4719 0; H -2.1407281956 -1.23595 0; H -2.1407281956 1.23595 0;"
)
elems, xyz = parse_molecule_string(benzene)

# 分子 3D 模型（沿用 flux_plot2 中的样式）
plot_molecule(elems, xyz, fig, ax2)

# 视角
view_elev = 40
view_azim = 40
ax2.view_init(elev=view_elev, azim=view_azim)

# 范围
ax2.set_xlim(-3.0, 3.0)
ax2.set_ylim(-3.0, 3.0)
ax2.set_zlim(-1.0, 1.0)
try:
    ax2.set_box_aspect([1, 1, 0.4])
except Exception:
    pass

# 去除网格与刻度
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])

# 去除 3D 框线和面（尽量“干净”）
ax2.set_axis_off()

# 简易三维坐标轴（用 Arrow3D + axis_kw）
origin = np.array([0.0, 0.0, 0.0])
axis_len = 4.0

arrow_x = Arrow3D([0, axis_len], [0, 0], [0, 0], **axis_kw)
arrow_y = Arrow3D([0, 0], [0, axis_len], [0, 0], **axis_kw)
arrow_z = Arrow3D([0, 0], [0, 0], [0, axis_len], **axis_kw)
ax2.add_artist(arrow_x)
ax2.add_artist(arrow_y)
ax2.add_artist(arrow_z)

ax2.text(axis_len+0.5, 0, 0, "x", fontsize=10, ha="center", va="center")
ax2.text(0, axis_len+0.5, 0, "y", fontsize=10, ha="center", va="center")
ax2.text(0, 0, axis_len+0.5, "z", fontsize=10, ha="center", va="center")

axis_kw2 = dict(
    arrowstyle='-|>',
    mutation_scale=2*axis_mutation_scale,
    lw=1.5,
    alpha=1.0,
    color='tab:blue',
    shrinkA=0,
    shrinkB=0
)
axis_kw3 = dict(
    arrowstyle='-|>',
    mutation_scale=2*axis_mutation_scale,
    lw=1.5,
    alpha=1.0,
    color='tab:orange',
    shrinkA=0,
    shrinkB=0
)
axis_kw4 = dict(
    arrowstyle='-|>',
    mutation_scale=2*axis_mutation_scale,
    lw=1.5,
    alpha=1.0,
    color='tab:green',
    shrinkA=0,
    shrinkB=0
)
# 跃迁偶极矩向量 µ_gx, µ_gy, µ_gz（示意长度，用同样 axis_kw）
mu_len = 3.5
mu_x = Arrow3D([0, mu_len], [0, 0], [0, 0],  **axis_kw2)
mu_y = Arrow3D([0, 0], [0, mu_len], [0, 0],  **axis_kw3)
mu_z = Arrow3D([0, 0], [0, 0], [0, mu_len/2],  **axis_kw4)
ax2.add_artist(mu_x)
ax2.add_artist(mu_y)
ax2.add_artist(mu_z)

ax2.text(mu_len*1.05, 0.5, 0.0,
         r"$\mu_{gX}$",
         fontsize=12, ha="left", va="center")
ax2.text(0, mu_len*1.05, 0.4,
         r"$\mu_{gY}$",
         fontsize=12, ha="center", va="bottom")
ax2.text(+0.1, -0.7, mu_len/2,
         r"$\mu_{gZ}$",
         fontsize=12, ha="center", va="bottom")

ax2.text(0, -2.9, 4, "(b)", fontsize=12, ha="left", va="top")

plt.tight_layout()
plt.show()