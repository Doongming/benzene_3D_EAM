# molecule_plot.py (fixed)
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from mpl_toolkits.mplot3d import Axes3D   

# ------------------------------------------------------------
# 1. 基础数据：颜色 & 共价半径（Å）
CPK_COLORS = {
    "H": "#FFFFFF", "C": "#000000", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
}
COVALENT_RADII = {             
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
}
DEFAULT_BOND_COLOR = "#dfe4e6"


@lru_cache(maxsize=None)
def _sphere_unit_mesh(res):
    """返回缓存的单位球网格，减少重复创建。"""
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    sin_v = np.sin(v)

    x = np.outer(np.cos(u), sin_v)
    y = np.outer(np.sin(u), sin_v)
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


@lru_cache(maxsize=None)
def _cylinder_unit_mesh(res_theta, res_len):
    """返回缓存的单位圆柱参数网格。"""
    theta = np.linspace(0.0, 2.0 * np.pi, res_theta)
    z = np.linspace(0.0, 1.0, res_len)
    theta_grid, z_grid = np.meshgrid(theta, z)
    sin_theta = np.sin(theta_grid)
    cos_theta = np.cos(theta_grid)
    return sin_theta, cos_theta, z_grid

def parse_molecule_string(s):
    elems, xyz = [], []
    for block in s.strip().split(";"):
        if block.strip():
            parts = block.split()
            if len(parts) != 4:
                raise ValueError(f"格式错误：{block}")
            elems.append(parts[0])
            xyz.append([float(x) for x in parts[1:]])
    return elems, np.asarray(xyz, dtype=float)

# ------------------------------------------------------------
def draw_sphere(ax, center, radius, color, res=28):
    """
    关闭抗锯齿、把边线颜色设为与面片同色，避免出现细的拼接线
    """
    x_unit, y_unit, z_unit = _sphere_unit_mesh(res)
    x = center[0] + radius * x_unit
    y = center[1] + radius * y_unit
    z = center[2] + radius * z_unit
    ax.plot_surface(
        x, y, z,
        color=color,
        shade=True,
        linewidth=0.0,
        edgecolors=color,     # 与面同色
        antialiased=False     # 关闭抗锯齿，去掉“细缝”
    )

def draw_cylinder(ax, p0, p1, radius, color, res_theta=20, res_len=4):
    """
    在 ax 上画端点 p0–p1 的圆柱（半径 radius）
    注意：这里假设 p0、p1 已经是“截断后”的位置（稍微伸进球体）
    """
    v = p1 - p0
    L = np.linalg.norm(v)
    if L <= 1e-6:
        return
    v = v / L
    # 找到与 v 不平行的一条向量
    not_v = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = np.cross(v, not_v); n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    sin_theta, cos_theta, z_norm = _cylinder_unit_mesh(res_theta, res_len)
    t = z_norm * L

    X = p0[0] + v[0]*t + radius*(sin_theta*n1[0] + cos_theta*n2[0])
    Y = p0[1] + v[1]*t + radius*(sin_theta*n1[1] + cos_theta*n2[1])
    Z = p0[2] + v[2]*t + radius*(sin_theta*n1[2] + cos_theta*n2[2])

    ax.plot_surface(
        X, Y, Z,
        color=color,
        shade=True,
        linewidth=0.0,
        edgecolors=color,     # 与面同色
        antialiased=False
    )

# ------------------------------------------------------------
def plot_molecule(elements, coords, fig, ax,
                  atom_scale=0.5, bond_radius=0.08,
                  bond_color=DEFAULT_BOND_COLOR, tolerance=0.4,
                  overlap_eps=0.02, auto_set_axes=True, axis_pad=2.0):
    """
    overlap_eps: 键端点向内“压进”球体的距离（Å），避免接口处出现缝隙/穿模
    """
    coords = np.asarray(coords, dtype=float)
    N = len(elements)

    # —— 可视化用球半径（与画球时一致）——
    vis_r = np.array([COVALENT_RADII.get(el, 0.7)*atom_scale for el in elements])
    cov_r = np.array([COVALENT_RADII.get(el, 0.7) for el in elements])

    if N == 0:
        return fig, ax

    # 先画化学键，后画球体（有助于遮挡正确）
    # ------ 1. 画化学键（“几何截断”到球面内少许） ------
    deltas = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(deltas, axis=-1)
    cov_threshold = cov_r[:, None] + cov_r[None, :] + tolerance
    bond_mask = (dists <= cov_threshold) & (dists > 1e-6)
    upper_idx = np.triu_indices(N, k=1)

    for i, j in zip(*upper_idx):
        if not bond_mask[i, j]:
            continue

        bond_length = dists[i, j]
        v = -deltas[i, j] / bond_length  # 单位方向：i -> j
        cut_i = max(vis_r[i] - overlap_eps, 0.0)
        cut_j = max(vis_r[j] - overlap_eps, 0.0)
        effective_length = bond_length - (cut_i + cut_j)
        if effective_length <= 1e-4:
            continue

        start = coords[i] + v * cut_i
        end = coords[j] - v * cut_j
        draw_cylinder(ax, start, end, bond_radius, bond_color)

    # ------ 2. 画原子球体（盖住接口）------
    for el, pos, r in zip(elements, coords, vis_r):
        c = CPK_COLORS.get(el, "#808080")
        draw_sphere(ax, pos, r, c)

    # ------ 3. 轴范围/外观 ------
    if auto_set_axes:
        pad = axis_pad + float(vis_r.max(initial=0.0))
        mins = coords.min(axis=0) - pad
        maxs = coords.max(axis=0) + pad
        ax.set_xlim3d(mins[0], maxs[0])
        ax.set_ylim3d(mins[1], maxs[1])
        ax.set_zlim3d(mins[2], maxs[2])
        set_axes_equal(ax)

    return fig, ax

def set_axes_equal(ax):
    """
    让 3D 坐标轴在视觉上等比例：
    基于当前 x/y/z 的 limits 拉到同一半径。
    """
    # 当前 limits
    xlim = np.array(ax.get_xlim3d(), dtype=float)
    ylim = np.array(ax.get_ylim3d(), dtype=float)
    zlim = np.array(ax.get_zlim3d(), dtype=float)

    # 统一半径
    ranges  = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
    centers = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius  = 0.5 * ranges.max()

    ax.set_xlim3d(centers[0]-radius, centers[0]+radius)
    ax.set_ylim3d(centers[1]-radius, centers[1]+radius)
    ax.set_zlim3d(centers[2]-radius, centers[2]+radius)

    # 新版 matplotlib 支持设置盒体等比例；旧版 try/except 忽略即可
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def set_axes_equal_from_data(ax, X, Y, Z, pad=0.05):
    """
    按数据范围设置等比例（推荐在你已经有 X/Y/Z 网格或采样点时使用）。
    pad 为相对留白比例。
    """
    mins   = np.array([np.nanmin(X), np.nanmin(Y), np.nanmin(Z)], dtype=float)
    maxs   = np.array([np.nanmax(X), np.nanmax(Y), np.nanmax(Z)], dtype=float)
    centers = (mins + maxs) / 2.0
    radius  = 0.5 * np.max(maxs - mins)
    radius *= (1.0 + pad)

    ax.set_xlim3d(centers[0]-radius, centers[0]+radius)
    ax.set_ylim3d(centers[1]-radius, centers[1]+radius)
    ax.set_zlim3d(centers[2]-radius, centers[2]+radius)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    
# ------------------------------------------------------------
if __name__ == "__main__":
    benzene = (
        "C 0 1.3964 0; C 1.2093178738 0.6982 0; C 1.2093178738 -0.6982 0;"
        "C 0 -1.3964 0; C -1.2093178738 -0.6982 0; C -1.2093178738 0.6982 0;"
        "H 0 2.4719 0; H 2.1407281956 1.23595 0; H 2.1407281956 -1.23595 0;"
        "H 0 -2.4719 0; H -2.1407281956 -1.23595 0; H -2.1407281956 1.23595 0;"
    )

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    elems, xyz = parse_molecule_string(benzene)
    plot_molecule(elems, xyz, fig, ax,
                  atom_scale=0.55,     # 你可以微调：球大一些更容易遮接口
                  bond_radius=0.10,    # 键粗一点观感更稳
                  tolerance=0.35,
                  overlap_eps=0.03)    # 多“压”一点更不容易露缝

    plt.tight_layout()
    plt.show()
