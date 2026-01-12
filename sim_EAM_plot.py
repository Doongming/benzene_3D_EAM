#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-level density-matrix simulator in LAB FRAME (no RWA) — all parameters in ATOMIC UNITS (a.u.)

Basis:
  |0> = |g>, |1> = |x>, |2> = |y>, |3> = |z|

Hamiltonian (ħ = 1 in a.u.):
  H(t) = H_m - μ · E(t)
  - H_m = diag(0, ω_x, ω_y, ω_z)                         [ENERGY in a.u. ≡ angular frequency in a.u.]
  - E_α(t) = ε_α * s(t) * cos(ω t + φ_α)                 [E-field ε_α in a.u.]
  - s(t) = exp[-t^2/(2 T^2)]                             [TIME t, T in a.u.]
  - μ_α are transition dipole moments in a.u. (e·a0)

Dissipation (Markovian, Lindblad):
  - Nonradiative decay from |α> to |g>: L_α = sqrt(γ_α) |g><α| with rate γ_α (a.u. of frequency)
  - Pure dephasing: for any i ≠ j, dρ_ij/dt ⊃ - γ_ij^(d) ρ_ij (a.u. of frequency)
    ⇒ total dephasing Γ_ij = 1/2(γ_i + γ_j) + γ_ij^(d)

ATOMIC-UNIT REMINDERS:
  - Energy: 1 a.u. (Hartree) = 27.211386245988 eV
  - Time:   1 a.u. = 2.4188843265857e-17 s
  - Ang. freq & rates: 1 a.u. = 4.1341373336493e16 s^-1
  - Dipole: 1 a.u. = e·a0 = 2.541746473 D
  - E-field: 1 a.u. = 5.14220674763e11 V/m
  - Intensity (for linear pol., plane wave): I[W/cm^2] ≈ 3.5094452e16 * (ε[a.u.])^2

This script has no CLI; run directly to execute a demo with physically plausible a.u. values.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# -------------------- Envelope (a.u.) --------------------
def gaussian_envelope(t,tc, T):
    """s(t) = exp[-(t-tc)^2/( T^2)], with t, T in atomic units of time."""
    return np.exp(-(t-tc)**2 / (  T**2))


# -------------------- Hamiltonian (lab frame, a.u.) --------------------
def build_H_lab(t, p):
    """
    Returns 4x4 complex Hamiltonian (a.u.).
    Basis order: 0=|g>, 1=|x>, 2=|y>, 3=|z|
    """
    # Carrier and transition angular frequencies (a.u.)
    w   = p["omega"]
    wx  = p["omega_x"]; wy = p["omega_y"]; wz = p["omega_z"]

    # Transition dipoles (a.u., e·a0)
    mux = p["mu_x"]; muy = p["mu_y"]; muz = p["mu_z"]

    # Electric-field peak amplitudes (a.u.)
    ex  = p["eps_x"]; ey = p["eps_y"]; ez = p["eps_z"]

    # CEP phases (radians)
    phx = p["phi_x"]; phy = p["phi_y"]; phz = p["phi_z"]

    # Pulse width (a.u. of time)
    T   = p["T"]
    tc  = p["tc"]

    s = gaussian_envelope(t,tc, T)
    Ex = ex * s * np.cos(w * t + phx)
    Ey = ey * s * np.cos(w * t + phy)
    Ez = ez * s * np.cos(w * t + phz)

    H = np.zeros((4, 4), dtype=complex)

    # Bare molecular energies (a.u.)
    H[1, 1] = wx
    H[2, 2] = wy
    H[3, 3] = wz

    # Dipole couplings g <-> α : V_gα(t) = - μ_α E_α(t) (a.u.)
    Vgx = - mux * Ex
    Vgy = - muy * Ey
    Vgz = - muz * Ez
    H[1, 0] += Vgx; H[0, 1] += Vgx
    H[2, 0] += Vgy; H[0, 2] += Vgy
    H[3, 0] += Vgz; H[0, 3] += Vgz

    return H


# -------------------- Dissipators (a.u.) --------------------
def lindblad_decay(rho, gx, gy, gz):
    """
    Nonradiative decay to |g>: rates gα in a.u. (s^-1 in SI divided by 4.1341e16).
    """
    D = np.zeros_like(rho, dtype=complex)
    Lx = np.zeros((4, 4), dtype=complex); Lx[0, 1] = 1.0
    Ly = np.zeros((4, 4), dtype=complex); Ly[0, 2] = 1.0
    Lz = np.zeros((4, 4), dtype=complex); Lz[0, 3] = 1.0

    for g, L in ((gx, Lx), (gy, Ly), (gz, Lz)):
        if g > 0.0:
            Lrho = L @ rho @ L.conj().T
            LdL  = L.conj().T @ L
            D += g * (Lrho - 0.5 * (LdL @ rho + rho @ LdL))
    return D


def pure_dephasing_term(rho, gamma_d):
    """
    Pure dephasing rates gamma_d[i,j] (a.u.) applied to off-diagonal elements ρ_ij, i≠j.
    """
    D = np.zeros_like(rho, dtype=complex)
    for i in range(4):
        for j in range(4):
            if i != j:
                gd = float(gamma_d[i, j])
                if gd > 0.0:
                    D[i, j] += - gd * rho[i, j]
    return D


# -------------------- Master equation RHS (a.u.) --------------------
def rhs_master_equation(t, y, p):
    rho = y.reshape((4, 4))
    H = build_H_lab(t, p)
    comm = -1j * (H @ rho - rho @ H)  # ħ = 1
    Ddec = lindblad_decay(rho, p["gamma_x"], p["gamma_y"], p["gamma_z"])
    Ddph = pure_dephasing_term(rho, p["gamma_d"])
    return (comm + Ddec + Ddph).reshape(-1)


# -------------------- Simulation (fixed settings) --------------------
def simulate(p, t_span, t_eval=None):
    """
    Integrate dρ/dt with fixed solver settings in a.u.
    """
    rho0 = np.zeros((4, 4), dtype=complex); rho0[0, 0] = 1.0  # start in |g>
    y0 = rho0.reshape(-1)
    sol = solve_ivp(lambda t, y: rhs_master_equation(t, y, p),
                    t_span, y0, t_eval=t_eval,
                    method="DOP853", rtol=1e-10, atol=1e-10)
    if not sol.success:
        raise RuntimeError("Integration failed: " + sol.message)
    rhos = sol.y.T.reshape((-1, 4, 4))
    return sol.t, rhos

# -------------------- Helpers --------------------
def populations(rhos):
    """Return populations [P_g, P_x, P_y, P_z]."""
    return np.real(np.stack([rhos[:, 0, 0], rhos[:, 1, 1], rhos[:, 2, 2], rhos[:, 3, 3]], axis=1))




lx0=6.064733e-02

ly0=6.066389e-02

lz0=5.886049e-01

def demo_params_au1():
    """
    Example parameters in atomic units (a.u.).
    - ω ~ 0.1 a.u. corresponds to photon energy ~ 2.72 eV (visible/UV border).
    - T = 1000 a.u. ~ 24.19 fs (so pulse FWHM ≈ 2.355 T ≈ 57 fs for Gaussian intensity;
      note we use s(t) on field — adjust to your convention if needed).
    - ε ~ 0.02 a.u. corresponds to intensity ~ 3.5e16*(0.02^2) ≈ 1.4e13 W/cm^2.
    - γ ~ 1e-4 a.u. ⇒ lifetime τ ~ 1/γ ≈ 1e4 a.u. ≈ 0.24 ps.
    """
    return {
        # Transition angular frequencies (a.u.) — nearly degenerate π-states
        "omega_x": 7.3448/27.211386245988,#7.3448/27.211386245988
        "omega_y": 7.3448/27.211386245988,   # 7.3448
        "omega_z": 7.1175/27.211386245988,#7.1175

        # Carrier angular frequency (a.u.)
        "omega":    (7.3448+7.1175)/2 /27.211386245988,# (7.3448+7.1175)/2

        # Transition dipoles (a.u., e·a0)
        "mu_x": 1.8952,
        "mu_y": 1.8952,
        "mu_z": 0.6149,

        # Electric-field peak amplitudes ε_α (a.u.)
        "eps_x":  0.0003295043621359726,
        "eps_y":  0.0003295943346415548,
        "eps_z":  0.0098565268752825,

        # Carrier–envelope phases φ_α (radians)
        "phi_x": 0.0,
        "phi_y": -np.pi/2,
        "phi_z": -np.pi/2,

        # Gaussian width T (a.u. of time)
        "T": 7*41.341373336493, #1 fs = 41.341373336493 au
        "tc": 0, #3*7*41.341373336493, #1 fs = 41.341373336493 au

        # Nonradiative decay rates γ_α (a.u.)
        "gamma_x": 0.01/27.211386245988,
        "gamma_y": 0.01/27.211386245988,
        "gamma_z": 0.01/27.211386245988,

        # Pure dephasing matrix γ_ij^(d) (a.u.), used only for i≠j. Order: 0=g,1=x,2=y,3=z
        "gamma_d": 0.01/27.211386245988*np.array([
            [0.0,    1,  1,  1],
            [1,   0.0,   1,  1],
            [1,   1,  0.0,   1],
            [1,   1,  1,  0.0 ]
        ], dtype=float),
    }


# -------------------- Demo (all a.u.) --------------------

p = demo_params_au1()

# Time window in a.u. (cover ±6σ of Gaussian field envelope)
T = p["T"]
tmin, tmax = -3* T, 15.0 * T
nt = 4801
t_eval = np.linspace(tmin, tmax, nt)

t, rhos = simulate(p, (tmin, tmax), t_eval=t_eval)
pops1 = populations(rhos)
s1 = gaussian_envelope(t, p['tc'], p['T'])


coh_xy1 = -2*np.imag(rhos[:, 1, 2])*lz0 #np.real(rhos[:, 1, 2]*1j*lz0-rhos[:, 2, 1]*1j*lz0)
coh_xz1 = -2*np.imag(rhos[:, 3, 1])*ly0 #np.real(rhos[:, 1, 3]*-1j*ly0+rhos[:, 3, 1]*1j*ly0)
coh_zy1 = -2*np.imag(rhos[:, 2, 3])*lx0  #np.real(rhos[:, 2, 3]*1j*lx0-rhos[:, 3, 2]*1j*lx0)
  


def demo_params_au2():
    """
    Example parameters in atomic units (a.u.).
    - ω ~ 0.1 a.u. corresponds to photon energy ~ 2.72 eV (visible/UV border).
    - T = 1000 a.u. ~ 24.19 fs (so pulse FWHM ≈ 2.355 T ≈ 57 fs for Gaussian intensity;
      note we use s(t) on field — adjust to your convention if needed).
    - ε ~ 0.02 a.u. corresponds to intensity ~ 3.5e16*(0.02^2) ≈ 1.4e13 W/cm^2.
    - γ ~ 1e-4 a.u. ⇒ lifetime τ ~ 1/γ ≈ 1e4 a.u. ≈ 0.24 ps.
    """
    return {
        # Transition angular frequencies (a.u.) — nearly degenerate π-states
        "omega_x": 7.3448/27.211386245988,#7.3448/27.211386245988
        "omega_y": 7.3448/27.211386245988,   # 7.3448
        "omega_z": 7.1175/27.211386245988,#7.1175

        # Carrier angular frequency (a.u.)
        "omega":    (7.3448+7.1175)/2 /27.211386245988,# (7.3448+7.1175)/2

        # Transition dipoles (a.u., e·a0)
        "mu_x": 1.8952,
        "mu_y": 1.8952,
        "mu_z": 0.6149,

        # Electric-field peak amplitudes ε_α (a.u.)
        "eps_x": 0., 
        "eps_y": 0.002285188086402198,
        "eps_z":  0.007043240301430226,

        # Carrier–envelope phases φ_α (radians)
        "phi_x": 0.0,
        "phi_y": 0.0,
        "phi_z": 0.0,

        # Gaussian width T (a.u. of time)
        "T": 7*41.341373336493, #1 fs = 41.341373336493 au
        "tc": 0. ,#3*7*41.341373336493, #1 fs = 41.341373336493 au

        # Nonradiative decay rates γ_α (a.u.)
        "gamma_x": 0.01/27.211386245988,
        "gamma_y": 0.01/27.211386245988,
        "gamma_z": 0.01/27.211386245988,

        # Pure dephasing matrix γ_ij^(d) (a.u.), used only for i≠j. Order: 0=g,1=x,2=y,3=z
        "gamma_d":  0.01/27.211386245988*np.array([
            [0.0,    1,  1,  1],
            [1,   0.0,   1,  1],
            [1,   1,  0.0,   1],
            [1,   1,  1,  0.0 ]
        ], dtype=float),
    }


# -------------------- Demo (all a.u.) --------------------
 
p = demo_params_au2()

# Time window in a.u. (cover ±6σ of Gaussian field envelope)
T = p["T"]
tmin, tmax = -3* T, 15.0 * T
nt = 4801
t_eval = np.linspace(tmin, tmax, nt)

t, rhos = simulate(p, (tmin, tmax), t_eval=t_eval)
pops2 = populations(rhos)
s2 = gaussian_envelope(t, p['tc'], p['T'])


coh_xy2 = -2*np.imag(rhos[:, 1, 2])*lz0 #np.real(rhos[:, 1, 2]*1j*lz0-rhos[:, 2, 1]*1j*lz0)
coh_xz2 = -2*np.imag(rhos[:, 3, 1])*ly0 #np.real(rhos[:, 1, 3]*-1j*ly0+rhos[:, 3, 1]*1j*ly0)
coh_zy2 = -2*np.imag(rhos[:, 2, 3])*lx0  #np.real(rhos[:, 2, 3]*1j*lx0-rhos[:, 3, 2]*1j*lx0)





def demo_params_au3():
    """
    Example parameters in atomic units (a.u.).
    - ω ~ 0.1 a.u. corresponds to photon energy ~ 2.72 eV (visible/UV border).
    - T = 1000 a.u. ~ 24.19 fs (so pulse FWHM ≈ 2.355 T ≈ 57 fs for Gaussian intensity;
      note we use s(t) on field — adjust to your convention if needed).
    - ε ~ 0.02 a.u. corresponds to intensity ~ 3.5e16*(0.02^2) ≈ 1.4e13 W/cm^2.
    - γ ~ 1e-4 a.u. ⇒ lifetime τ ~ 1/γ ≈ 1e4 a.u. ≈ 0.24 ps.
    """
    return {
        # Transition angular frequencies (a.u.) — nearly degenerate π-states
        "omega_x": 7.3448/27.211386245988,#7.3448/27.211386245988
        "omega_y": 7.3448/27.211386245988,   # 7.3448
        "omega_z": 7.1175/27.211386245988,#7.1175

        # Carrier angular frequency (a.u.)
        "omega":    (7.3448+7.1175)/2 /27.211386245988,# (7.3448+7.1175)/2

        # Transition dipoles (a.u., e·a0)
        "mu_x": 1.8952,
        "mu_y": 1.8952,
        "mu_z": 0.6149,

        # Electric-field peak amplitudes ε_α (a.u.)
        "eps_x": 0.0022851880,  
        "eps_y": 0.0022851880,
        "eps_z":  0,

        # Carrier–envelope phases φ_α (radians)
        "phi_x": 0.0,
        "phi_y": -np.pi/2,
        "phi_z": 0.0,

        # Gaussian width T (a.u. of time)
        "T": 7*41.341373336493, #1 fs = 41.341373336493 au
        "tc": 0.,# 3*7*41.341373336493, #1 fs = 41.341373336493 au

        # Nonradiative decay rates γ_α (a.u.)
        "gamma_x": 0.01/27.211386245988,
        "gamma_y": 0.01/27.211386245988,
        "gamma_z": 0.01/27.211386245988,

        # Pure dephasing matrix γ_ij^(d) (a.u.), used only for i≠j. Order: 0=g,1=x,2=y,3=z
        "gamma_d": 0.01/27.211386245988*np.array([
            [0.0,    1,  1,  1],
            [1,   0.0,   1,  1],
            [1,   1,  0.0,   1],
            [1,   1,  1,  0.0 ]
        ], dtype=float),
    }


# -------------------- Demo (all a.u.) --------------------

p = demo_params_au3()

# Time window in a.u. (cover ±6σ of Gaussian field envelope)
T = p["T"]
tmin, tmax = -3* T, 15.0 * T
nt = 4801
t_eval = np.linspace(tmin, tmax, nt)

t, rhos = simulate(p, (tmin, tmax), t_eval=t_eval)
pops3 = populations(rhos)
s3 = gaussian_envelope(t, p['tc'], p['T'])

coh_xy3 = np.real(rhos[:, 1, 2]*1j*lz0-rhos[:, 2, 1]*1j*lz0)
coh_xz3 = np.real(rhos[:, 1, 3]*-1j*ly0+rhos[:, 3, 1]*1j*ly0)
coh_zy3 = np.real(rhos[:, 2, 3]*1j*lx0-rhos[:, 3, 2]*1j*lx0)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d  # 投影到 2D

# -------- 3D Arrow patch（matplotlib.patches 家族）--------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = (np.asarray(xs), np.asarray(ys), np.asarray(zs))

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
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))

tmp = [2.5,2.2,1.5]
def plot_pra_6panels_with_L3D(
    datasets,
    xunit="fs",
    normalize_envelope=True,
    figsize=(10.0, 7.8),
    spine_lw=0.3,   # 边框粗细
    tick_lw=0.3,    # 刻度粗细
    tick_len=3.0,   # 刻度长度
    select_range=(21, 39.0),
    n_vectors=10,
    cmap="Reds",                
    cmap_reverse=False,         
    cmap_range=(0.25, 0.95),    
    vec_mutation_scale=8,       
    axis_mutation_scale=7,      
    alpha_start=0.2,            
    alpha_end=1.0,              
    view_elev=40,
    view_azim=40,
    wspace=0.01, # 稍微增加一点间距给右侧刻度留位置
    show=True
):
    """
    修改版：在中间列添加解析解（双Y轴，半透明，颜色对应）。
    """
    assert len(datasets) == 3, "datasets 应包含 3 组参数（3 行）。"

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": spine_lw,
    })

    # --- 辅助：计算解析解数据 ---
    def _get_theoretical_data(row_idx, t_fs):
        """
        根据用户提供的公式返回 (Lx, Ly, Lz) 的理论值
        Case A (row 0): Lz = 0.59
        Case B (row 1): Lx = 0.06 * sin(dw * t)
        Case C (row 2): Lx = L_bar*cos(dw*t), Ly = L_bar*sin(-dw*t), Lz = L_bar
        """
        # 物理常数
        hbar_eVfs = 0.6582119569 
        dE_eV = 0.2273
        dw = dE_eV / hbar_eVfs # rad/fs approx 0.349
        
        Lx_th, Ly_th, Lz_th = None, None, None
        
        if row_idx == 0: # Case A
            Lz_const = 5.886049e-01
            # 生成常数数组
            Lz_th = np.full_like(t_fs, Lz_const)
            # Lx, Ly 为 0，通常不需要画，或者画零线
            
        elif row_idx == 1: # Case B
            Lx_amp = 6.064733e-02
            Lx_th = Lx_amp * np.sin(dw * (t_fs-0))
            # Ly, Lz 为 0
            
        elif row_idx == 2: # Case C
            L_bar = 1.22e-2
            Lx_th = L_bar * np.sin(dw * (t_fs-0)+0)
            Ly_th = L_bar * np.sin(-dw * (t_fs-0)-np.pi/2) # sin(-x) = -sin(x)
            Lz_th = np.full_like(t_fs, L_bar)
            
        return Lx_th, Ly_th, Lz_th

    # 内部样式设置函数
    def _set_thin_axes(ax, is_twin=False):
        for side in ["left", "right", "top", "bottom"]:
            if side in ax.spines:
                ax.spines[side].set_linewidth(spine_lw)
        # 刻度设置
        ax.tick_params(axis="both", which="both", width=tick_lw, length=tick_len, direction="in")
        return ax

    def _auto_ylim(arrs, pad=0.3):
        finite = []
        for a in arrs:
            if a is None: continue
            aa = np.asarray(a)
            if aa.size == 0: continue
            finite.append(aa[np.isfinite(aa)])
        if not finite: return (0.0, 1.0)
        vals = np.hstack(finite)
        if vals.size == 0: return (0.0, 1.0)
        ymin, ymax = float(np.min(vals)), float(np.max(vals))
        span = ymax - ymin
        if span < 1e-12: span = 1.0
        return ymin - pad*span, ymax + pad*span

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, wspace=wspace, hspace=0.01)
    xlab = "Time (fs)" if xunit.lower() == "fs" else "Time (a.u.)"

    rho_styles = [
        dict(color="black",      linestyle="-",  linewidth=1., label=r"$\rho_{gg}$"),
        dict(color="tab:blue",   linestyle="-",  linewidth=1., label=r"$\rho_{XX}$"),
        dict(color="tab:orange", linestyle="--", linewidth=1., label=r"$\rho_{YY}$"),
        dict(color="tab:green",  linestyle="-.", linewidth=1., label=r"$\rho_{ZZ}$"),
    ]
    # 模拟结果样式
    L_styles = [
        dict(color="blue",    linestyle="-",  linewidth=1., label=r"$\langle L_x\rangle$"),
        dict(color="red", linestyle="--", linewidth=1., label=r"$\langle L_y \rangle$"),
        dict(color="green",  linestyle="-.", linewidth=1., label=r"$\langle L_z \rangle$"),
    ]
    
    # 理论结果样式 (增加透明度 alpha, 线宽稍大以显示在背后)
    L_styles_th = [
        dict(color="blue",    linestyle="-",  linewidth=1, alpha=0.25),
        dict(color="red", linestyle="-", linewidth=1, alpha=0.25),
        dict(color="green",  linestyle="-", linewidth=1, alpha=0.25),
    ]

    axs = []
    for i in range(3):
        axL = fig.add_subplot(gs[i, 0])
        axM = fig.add_subplot(gs[i, 1])
        axR = fig.add_subplot(gs[i, 2], projection="3d")
        axs.append([axL, axM, axR])

    cmap_obj = plt.get_cmap(cmap)
    c0, c1 = cmap_range
    cvals = np.linspace(c0, c1, n_vectors)
    if cmap_reverse: cvals = cvals[::-1]

    # X轴刻度
    xticks_val = [-20,0, 20, 40, 65,80,100]

    for i, data in enumerate(datasets):
        t   = np.asarray(data["t"])
        # 数据解包...
        r00, rxx, ryy, rzz = data["rho_00"], data["rho_xx"], data["rho_yy"], data["rho_zz"]
        Lx, Ly, Lz = data["Lx"], data["Ly"], data["Lz"]
        env = data["envelope"]

        # ===== 左列：ρ + envelope =====
        axL = axs[i][0]; _set_thin_axes(axL)
        linesL = []
        linesL += [axL.plot(t, r00, **rho_styles[0])[0]]
        linesL += [axL.plot(t, rxx, **rho_styles[1])[0]]
        linesL += [axL.plot(t, ryy, **rho_styles[2])[0]]
        linesL += [axL.plot(t, rzz, **rho_styles[3])[0]]

        axL.set_xlim(np.nanmin(t), np.nanmax(t)); axL.margins(x=0)
        axL.set_xticks(xticks_val)
        if i < 2: axL.tick_params(labelbottom=False)

        ymin, ymax = _auto_ylim([r00, rxx, ryy, rzz])
        axL.set_ylim(max(0.0, ymin), max(1.0, ymax + 0.1))
        axL.set_ylabel(r"Population")
        if i == 2: axL.set_xlabel(xlab)

        # Envelope
        if normalize_envelope and np.max(np.abs(env)) > 0:
            env = env / np.max(np.abs(env))
        axR2 = axL.twinx(); _set_thin_axes(axR2); axR2.set_yticks([])
        for spine in axR2.spines.values(): spine.set_visible(False)
        axR2.set_ylim(0.0, 1.05)
        # axR2.fill_between(t, 0, env, color='gray', alpha=0.1) # 可选：填充包络
        
        axL.legend(
            handles=linesL,
            loc="upper right",
            frameon=False,
            ncol=2,
            # --- 新增/调整的参数 ---
            columnspacing=0.5,   # 默认约为 2.0，改小可以拉近列间距
            handletextpad=0.2,   # 默认约为 0.8，改小可以让文字紧贴线条
            borderaxespad=0.2    # 减少图例与坐标轴边界的留白
        )
        # ===== 中列：Lx/Ly/Lz (Sim + Theory) =====
        axM = axs[i][1]; _set_thin_axes(axM)
        
        # 1. 绘制模拟曲线 (左轴)
        linesM = []
        linesM += [axM.plot(t, Lx, **L_styles[0])[0]]
        linesM += [axM.plot(t, Ly, **L_styles[1])[0]]
        linesM += [axM.plot(t, Lz, **L_styles[2])[0]]

        axM.set_xlim(np.nanmin(t), np.nanmax(t)); axM.margins(x=0)
        axM.set_xticks(xticks_val)
        if i < 2: axM.tick_params(labelbottom=False)

        # 自动调整左轴范围
        yminM, ymaxM = _auto_ylim([Lx, Ly, Lz])
        # 保证0在视野中
        if yminM > 0: yminM = -0.2 * ymaxM
        if ymaxM < 0: ymaxM = -0.2 * yminM
        axM.set_ylim(yminM, ymaxM)
        axM.set_ylabel(r"Sim. EAM (a.u.)")
        if i == 2: axM.set_xlabel(xlab)

        # 2. 绘制解析曲线 (右轴 Twinx)
        axM_th = axM.twinx()
        _set_thin_axes(axM_th, is_twin=True) # 应用细边框
        
        Lx_th, Ly_th, Lz_th = _get_theoretical_data(i, t)
        
        # 仅绘制非 None 的理论线
        if Lx_th is not None: axM_th.plot(t[1600:], Lx_th[1600:], **L_styles_th[0])
        if Ly_th is not None: axM_th.plot(t[1600:], Ly_th[1600:], **L_styles_th[1])
        if Lz_th is not None: axM_th.plot(t[1600:], Lz_th[1600:], **L_styles_th[2])
        
        # 自动调整右轴范围
        axM_th.set_ylim(yminM*tmp[i], ymaxM*tmp[i])
        axM_th.set_ylabel(r"Theory (a.u.)", color='gray', fontsize=8)
        axM_th.tick_params(axis='y', colors='gray', labelcolor='gray')

        # 辅助线
        axM.axvline(21, linestyle="--", color="0.5", linewidth=0.8)
        if xunit.lower() == "fs":
            axM.axvline(39.0, linestyle="--", color="0.5", linewidth=0.8)

        # Legend (只显示模拟的图例即可，颜色已对应)
        axM.legend(handles=linesM, loc="upper right", frameon=False, ncol=3, handlelength=1.5,
                   columnspacing=0.5,   # 默认约为 2.0，改小可以拉近列间距
                    handletextpad=0.2,   # 默认约为 0.8，改小可以让文字紧贴线条
                    borderaxespad=0.2)    # 减少图例与坐标轴边界的留白)

        # ===== 右列：3D L 向量 =====
        ax3 = axs[i][2]
        # (3D 绘图逻辑保持不变...)
        tmin_sel, tmax_sel = select_range
        mask = (t >= tmin_sel) & (t <= tmax_sel)
        idx_pool = np.where(mask)[0]
        if idx_pool.size == 0: idx_pool = np.arange(t.size)
        if idx_pool.size >= n_vectors:
            ii = np.linspace(0, idx_pool.size - 1, n_vectors, dtype=int)
            idx = idx_pool[ii]
        else: idx = idx_pool

        alphas = np.linspace(alpha_start, alpha_end, max(1, idx.size))
        Lx_sel, Ly_sel, Lz_sel = Lx[idx], Ly[idx], Lz[idx]
        norms = np.sqrt(Lx_sel**2 + Ly_sel**2 + Lz_sel**2)
        maxnorm = np.nanmax(norms) if np.isfinite(norms).any() else 1.0
        if maxnorm == 0: maxnorm = 1.0
        lim = 1.1 * maxnorm

        ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
        ax3.view_init(elev=view_elev, azim=view_azim)
        try: ax3.set_box_aspect([1, 1, 1])
        except: pass
        ax3.set_axis_off()
        ax3.scatter([0], [0], [0], s=5, c="black", depthshade=False, zorder=10)

        # 坐标轴
        axis_len = lim * 1
        axis_kw = dict(arrowstyle='-|>', mutation_scale=axis_mutation_scale, lw=spine_lw, color='0.0')
        ax3.add_artist(Arrow3D([0, axis_len], [0, 0], [0, 0], **axis_kw))
        ax3.add_artist(Arrow3D([0, 0], [0, axis_len], [0, 0], **axis_kw))
        ax3.add_artist(Arrow3D([0, 0], [0, 0], [0, axis_len], **axis_kw))
        ax3.text(axis_len*1.2, 0, 0, "x", color='black', fontsize=8)
        ax3.text(0, axis_len*1.03, 0, "y", color='black', fontsize=8)
        ax3.text(0, 0, axis_len*1.03, "z", color='black', fontsize=8)

        # 数据箭头
        vec_kw = dict(arrowstyle='-|>', mutation_scale=vec_mutation_scale, lw=1.2)
        n_ar = len(idx)
        cvals_used = np.interp(np.linspace(0, 1, n_ar), np.linspace(0, 1, len(cvals)), cvals)
        for j, k in enumerate(idx):
            rgba = list(cmap_obj(cvals_used[j]))
            rgba[3] = float(alphas[j])
            ax3.add_artist(Arrow3D([0, float(Lx[k])], [0, float(Ly[k])], [0, float(Lz[k])], color=tuple(rgba), **vec_kw))

    # Labels
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    for idx_row, ax_row in enumerate(axs):
        for idx_col, ax in enumerate(ax_row):
            label = panel_labels[idx_row * 3 + idx_col]
            if idx_col == 2: ax.text2D(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=9)
            else: ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=9)

    if show: plt.show()
    return fig, np.array(axs)

# ======== 数据集定义 (保持您的顺序：Case A, Case B, Case C) ========
# 注意：datasets[0]对应Case A, datasets[1]对应Case B, datasets[2]对应Case C
datasets = [
    {   # Row 0: Case A (pops3 -> eps_z=0, circular XY)
        "t": t/41.341373336493,
        "rho_00": pops3[:, 0], "rho_xx": pops3[:, 1], "rho_yy": pops3[:, 2], "rho_zz": pops3[:, 3],
        "Lx": coh_zy3, "Ly": coh_xz3, "Lz": coh_xy3, "envelope": s3,
    },
    {   # Row 1: Case B (pops2 -> eps_y=0, X+Z linear)
        "t": t/41.341373336493,
        "rho_00": pops2[:, 0], "rho_xx": pops2[:, 1], "rho_yy": pops2[:, 2], "rho_zz": pops2[:, 3],
        "Lx": coh_zy2, "Ly": coh_xz2, "Lz": coh_xy2, "envelope": s2,
    },
    {   # Row 2: Case C (pops1 -> all on, conical)
        "t": t/41.341373336493,
        "rho_00": pops1[:, 0], "rho_xx": pops1[:, 1], "rho_yy": pops1[:, 2], "rho_zz": pops1[:, 3],
        "Lx": coh_zy1, "Ly": coh_xz1, "Lz": coh_xy1, # 注意这里保持您原本的负号
        "envelope": s1,
    },
]

plot_pra_6panels_with_L3D(datasets, xunit="fs", normalize_envelope=True)