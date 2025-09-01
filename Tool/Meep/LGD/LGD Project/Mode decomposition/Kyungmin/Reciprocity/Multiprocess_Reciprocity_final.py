# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 저장 안정화
import matplotlib.pyplot as plt

import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa
from autograd import tensor_jacobian_product
from meep.materials import Ag

# ===================== 파라미터 =====================
mp.verbosity(0)

lambda_min, lambda_max = 0.4, 0.7
fmin, fmax = 1/lambda_max, 1/lambda_min
fcen, df   = 0.5*(fmin+fmax), (fmax - fmin)
nfreq      = 1

resolution = 50
design_region_resolution = 50

# 층/지오메트리
tag_bot, tag_side, tsio2 = 0.5, 4.0, 4.0
tpml = 0.4
monitor_position, monitor_blank = 1.0, 0.05

width_ag, width_sio2 = 0.2, 2.0
width_bot_ag = round(width_ag + width_sio2 + width_ag, 2)
design_region_width, design_region_height = round(width_bot_ag, 3), 0.5

Sx = tpml + width_bot_ag + tpml
Sy = round(tpml + tag_bot + tag_side + design_region_height + monitor_position + monitor_blank + tpml, 3)
cell_size = mp.Vector3(Sx, Sy)

SiO2 = mp.Medium(index=1.45)
TiO2 = mp.Medium(index=2.6)
Air  = mp.Medium(index=1.0)

DESIGN_MODE = 'free'     # 'free' 또는 'grating'
Nx = int(design_region_resolution * design_region_width) + 1
Ny = 1 if DESIGN_MODE == 'grating' else int(design_region_resolution * design_region_height) + 1
layer_num = 1  # 단일 레이어로 단순화

# 위치
y_bottom = -Sy/2
center_y_ag     = y_bottom + tpml + tag_bot/2
center_y_sio2   = y_bottom + tpml + tag_bot + tsio2/2
center_y_design = y_bottom + tpml + tag_bot + tsio2 + design_region_height/2
center_y_monitor_position = y_bottom + tpml + tag_bot + tsio2 + design_region_height + monitor_position
center_y_source_position  = y_bottom + tpml + tag_bot + 1/fcen
center_Ag_x_position = design_region_width/2 - width_ag/2

boundary_layers = [mp.Absorber(tpml, direction=mp.X),
                   mp.PML(tpml, direction=mp.Y)]

# ===================== 지오메트리 =====================
geometry = [
    mp.Block(material = Ag,   size=mp.Vector3(width_bot_ag, tag_bot, 0), center=mp.Vector3(0, center_y_ag, 0)),
    mp.Block(material = SiO2, size=mp.Vector3(width_sio2,   tsio2,   0), center=mp.Vector3(0, center_y_sio2, 0)),
    mp.Block(material = Ag,   size=mp.Vector3(width_ag,     tsio2,   0), center=mp.Vector3(-center_Ag_x_position, center_y_sio2, 0)),
    mp.Block(material = Ag,   size=mp.Vector3(width_ag,     tsio2,   0), center=mp.Vector3( center_Ag_x_position, center_y_sio2, 0)),
]

# Design region (단일 레이어)
mg = mp.MaterialGrid(mp.Vector3(Nx, Ny), TiO2, SiO2, grid_type="U_MEAN", do_averaging=False)
design_region = mpa.DesignRegion(
    mg, volume=mp.Volume(center=mp.Vector3(0, center_y_design, 0),
                         size=mp.Vector3(design_region_width, design_region_height, 0))
)
geometry.append(mp.Block(material=mg, size=design_region.size, center=design_region.center))

# ===================== 소스/시뮬레이션 =====================
num_sources = 1
source_x_position = 0.0
spacing = 10 / resolution
offsets = [ (i - (num_sources - 1) / 2) * spacing for i in range(num_sources) ]

def make_source(xc):
    return mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(xc, center_y_source_position, 0)  # ← 소스는 source 위치
    )

sources = [make_source(source_x_position + off) for off in offsets]

sims = [
    mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=[src],
        default_material=Air,
        extra_materials=[Ag],
    ) for src in sources
]

# ===================== 모니터(FourierFields) =====================
monitor_center = mp.Vector3(0, center_y_monitor_position, 0)    # ← 모니터는 monitor 위치
monitor_size   = mp.Vector3(width_sio2 - 2/resolution, 0)

FFs = [mpa.FourierFields(sim, mp.Volume(center=monitor_center, size=monitor_size),
                         mp.Ez, yee_grid=True)
       for sim in sims]

J = lambda fields: npa.sum(npa.abs(fields)**2, axis=1)

opt_list = [
    mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J],
        objective_arguments=[FFs[i]],
        design_regions=[design_region],
        frequencies=[fcen],
        maximum_run_time=100,
    ) for i, sim in enumerate(sims)
]

# ===================== 출력/그림 저장 유틸 =====================
OUT_DIR = f"ndipole{num_sources}_resolution{resolution}_{DESIGN_MODE}_loc{source_x_position:.2f}_final"
os.makedirs(OUT_DIR, exist_ok=True)

def save_figure(path, fig=None, dpi=220):
    if fig is None: fig = plt.gcf()
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_sim_geometry(sim, filename, show_src=False, show_mon=False):
    fig, ax = plt.subplots(figsize=(6,6), dpi=220)
    sim.plot2D(ax=ax, plot_geometry=True, plot_boundaries=True,
               plot_sources=show_src, plot_monitors=show_mon)
    ax.set_xlim(-Sx/2, Sx/2); ax.set_ylim(-Sy/2, Sy/2)
    ax.set_aspect('equal', adjustable='box')
    save_figure(os.path.join(OUT_DIR, filename), fig)

# 초기 구조 저장 (각 소스별 동일 지오메트리)
for i in range(num_sources):
    save_sim_geometry(opt_list[i].sim, f"init_geometry_src{i+1}.png", show_src=True, show_mon=True)

# ===================== 최적화 준비 =====================
class AdamOptimizer:
    def __init__(self, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, warmup_iters=10):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.mt = None; self.vt = None; self.t = 0; self.warmup_iters = warmup_iters
        self.base_lr = lr
    def update(self, v, g):
        if self.mt is None: self.mt = np.zeros_like(v)
        if self.vt is None: self.vt = np.zeros_like(v)
        self.t += 1
        self.mt = self.b1*self.mt + (1-self.b1)*g
        self.vt = self.b2*self.vt + (1-self.b2)*(g**2)
        mhat = self.mt/(1-self.b1**self.t)
        vhat = self.vt/(1-self.b2**self.t)
        lr = self.lr * (self.t/self.warmup_iters) if self.t <= self.warmup_iters else self.lr
        step = lr * mhat/(np.sqrt(vhat)+self.eps)
        return np.clip(v + step, 0.0, 1.0)

eta_i = 0.5
def mapping_grating(x, eta, beta):
    layer = x.reshape(Nx, Ny)
    proj  = mpa.tanh_projection(layer, beta, eta)
    return proj.ravel()
def mapping_free(x, eta, beta):
    field = x.reshape(Nx, Ny)
    proj  = mpa.tanh_projection(field, beta, eta)
    return proj.ravel()

mapping_fn = mapping_grating if DESIGN_MODE == 'grating' else mapping_free
print(f"[Design Mode] {DESIGN_MODE} (Nx={Nx}, Ny={Ny})")

def update_beta(evals, bins, beta):
    if len(evals) >= 3:
        m2, m1, m0 = map(np.mean, evals[-3:])
        b2, b1, b0 = bins[-3:]
        c1 = abs(m0-m1)/(abs(m1)+1e-12)
        c2 = abs(m1-m2)/(abs(m2)+1e-12)
        bc1= abs(b0-b1)/(abs(b1)+1e-12)
        bc2= abs(b1-b2)/(abs(b2)+1e-12)
        if (c1<5e-3 and c2<5e-3) and (bc1<1e-3 and bc2<1e-3):
            beta *= 1.2
    return beta

n = Nx*Ny
np.random.seed(4)
x = 0.4 + 0.2*np.random.rand(n)
cur_beta = 4.0

opt = AdamOptimizer(lr=0.02)

evaluation_history, beta_history, binarization_history = [], [], []
Max_iter = 300

# ===================== 최적화 루프 =====================
def step_once(v, eta, beta):
    rho = mapping_fn(v, eta, beta)
    # 이진화도
    bin_deg = np.sum(np.abs(rho - 0.5)) / (0.5 * rho.size)
    binarization_history.append(bin_deg)

    # FoM, dJ/drho
    f_vals, dJ_list = [], []
    for OP in opt_list:
        f0, dJ = OP([rho], need_value=True, need_gradient=True, beta=beta)
        f_vals.append(float(np.abs(f0).item()))
        dJ_list.append(dJ)
    evaluation_history.append(f_vals)
    dJ_mean = np.mean(dJ_list, axis=0).flatten()

    # 체인룰: drho/dv 적용
    g_full = tensor_jacobian_product(mapping_fn, 0)(v, eta, beta, dJ_mean)
    g_full = g_full / (np.linalg.norm(g_full.ravel())/np.sqrt(g_full.size)+1e-12)

    v_new = opt.update(v, g_full)
    beta_history.append(beta)
    return v_new

for it in range(1, Max_iter+1):
    x = step_once(x, eta_i, cur_beta)
    cur_beta = update_beta(evaluation_history, binarization_history, cur_beta)  # ← 실제 반영
    if binarization_history[-1] > 0.98:
        print(f"Stop at iter {it}: binarization {binarization_history[-1]:.3f}")
        break

# ===================== 결과 저장 =====================
eval_arr = np.array(evaluation_history)
its = np.arange(len(evaluation_history))

# [1] 각 소스 FoM
fig1 = plt.figure(figsize=(6,6), dpi=220)
for i in range(eval_arr.shape[1]):
    plt.plot(its, eval_arr[:, i], label=f"Source {i+1}")
plt.xlabel('Iteration'); plt.ylabel('FoM (per source)')
plt.grid(True); plt.legend()
save_figure(os.path.join(OUT_DIR, "fom_per_source.png"), fig1)

# [2] 평균 FoM 정규화
fig2 = plt.figure(figsize=(6,6), dpi=220)
mean_fom = eval_arr.mean(axis=1)
plt.plot(its, mean_fom/np.max(mean_fom))
plt.xlabel('Iteration'); plt.ylabel('Normalized FoM'); plt.grid(True)
save_figure(os.path.join(OUT_DIR, "fom_mean.png"), fig2)

# [3] Beta & Binarization
fig3, ax1 = plt.subplots(figsize=(6,6), dpi=220)
ax1.set_xlabel('Iteration'); ax1.set_ylabel('Beta', color='tab:red')
ax1.plot(beta_history, color='tab:red'); ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.grid(True)
ax2 = ax1.twinx()
ax2.set_ylabel('Binarization', color='tab:green'); ax2.plot(binarization_history, color='tab:green'); ax2.set_ylim(0,1)
save_figure(os.path.join(OUT_DIR, "beta_binarization.png"), fig3)

# 최종 구조 저장 (rho를 sim에 반영 후 렌더)
rho_final = mapping_fn(x, eta_i, cur_beta)
_ = opt_list[0]([rho_final], need_value=False, need_gradient=False, beta=cur_beta)
save_sim_geometry(opt_list[0].sim, "final_geometry.png", show_src=True, show_mon=True)

# 디자인 변수(현재 mg.weights) 보관
np.savetxt(os.path.join(OUT_DIR, "design_weights.txt"), mg.weights)

print("Saved to:", os.path.abspath(OUT_DIR))
