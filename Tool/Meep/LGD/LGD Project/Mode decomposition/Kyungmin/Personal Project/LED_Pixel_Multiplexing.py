import meep as mp
import meep.adjoint as mpa
from meep.materials import Ag
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product

# --- Optimizer 설정 ---
class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, warmup_iters=10):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mt = None
        self.vt = None
        self.iter = 0
        self.warmup_iters = warmup_iters
        
    def reset_momenta(self):
        self.mt = None
        self.vt = None
        self.iter = 0
        
    def update(self, v, gradient):
        if self.mt is None:
            self.mt = np.zeros_like(v)
        if self.vt is None:
            self.vt = np.zeros_like(v)
        self.iter += 1
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * gradient
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.mt / (1 - self.beta1 ** self.iter)
        v_hat = self.vt / (1 - self.beta2 ** self.iter)
        if self.iter <= self.warmup_iters:
            warmup_factor = self.iter / self.warmup_iters
            lr = self.lr * warmup_factor
        else:
            lr = self.lr
        update_factor = m_hat / (np.sqrt(v_hat) + self.epsilon)
        update = lr * update_factor
        updated_v = np.clip(v + update, 0.0, 1.0)
        adam_lr=np.mean(np.abs(update))
        adam_uf=np.mean(np.abs(update_factor))
        adam_m=self.mt; adam_v=self.vt; adam_t=self.iter
        return updated_v, adam_lr, adam_uf, adam_m, adam_v, adam_t

# --- 기본 설정 ---
mp.verbosity(0)
resolution = 50
tpml = 0.5
boundary_layers = [mp.PML(thickness=tpml, direction=mp.Y)]
lambda_min = 0.4
lambda_max = 0.7

fmax = 1/lambda_min
fmin = 1/lambda_max

fcen = 0.5*(fmin+fmax)
df = fmax-fmin

# --- 재료 ---
SiO2  = mp.Medium(index=1.47)
air   = mp.Medium(index=1.0)
gan   = mp.Medium(index=2.48)
Al2O3 = mp.Medium(index=1.78)

# --- 기하 파라미터 ---
ag_bottom_x = 1.00
ag_bottom_y = 0.10
gan_y       = 1.00
SiO2_x      = 0.07
Al2O3_x     = 0.07
ag_side_x   = 0.10

side_stack_y = ag_bottom_y + gan_y          # 1.10
design_region_width  = 1.14
design_region_height = 0.15

# --- 셀 크기 (내부 디바이스 + PML) ---
device_width  = ag_bottom_x + 2*(SiO2_x + Al2O3_x + ag_side_x)
device_height = ag_bottom_y + gan_y + design_region_height
Sx = device_width*2
Sy = device_height + 2*tpml + 2/fcen
cell_size = mp.Vector3(Sx, Sy, 0)

# --- 내부 영역 엣지 ---
y_min_in = -Sy/2 + tpml

# --- 수평 센터 위치 ---
ag_lside_cx  = -(ag_bottom_x/2 + SiO2_x + Al2O3_x + ag_side_x/2)
ag_rside_cx  = +(ag_bottom_x/2 + SiO2_x + Al2O3_x + ag_side_x/2)
SiO2_lside_cx  = -(ag_bottom_x/2 + SiO2_x/2)
SiO2_rside_cx  = +(ag_bottom_x/2 + SiO2_x/2)
Al2O3_lside_cx = -(ag_bottom_x/2 + SiO2_x + Al2O3_x/2)
Al2O3_rside_cx = +(ag_bottom_x/2 + SiO2_x + Al2O3_x/2)

# --- 수직 센터 위치 ---
ag_bottom_cy = y_min_in + ag_bottom_y/2
gan_cy       = y_min_in + ag_bottom_y + gan_y/2
side_stack_cy = y_min_in + side_stack_y/2
design_cy    = y_min_in + ag_bottom_y + gan_y + design_region_height/2

# --- 디자인 그리드 ---
Nx = int(round(resolution * design_region_width)) + 1
Ny = int(round(resolution * design_region_height)) + 1
design_variables = mp.MaterialGrid(
    mp.Vector3(Nx, Ny),
    air, SiO2,
    grid_type="U_MEAN",
    do_averaging=False
)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(0, design_cy, 0),
        size=mp.Vector3(design_region_width, design_region_height, 0),
    ),
)

# --- 기하 ---
geometry = [
    mp.Block(center=mp.Vector3(0, gan_cy, 0),
             material=gan,
             size=mp.Vector3(ag_bottom_x, gan_y, 0)),
    mp.Block(center=mp.Vector3(ag_lside_cx, side_stack_cy, 0),
             material=Ag,
             size=mp.Vector3(ag_side_x, side_stack_y, 0)),
    mp.Block(center=mp.Vector3(ag_rside_cx, side_stack_cy, 0),
             material=Ag,
             size=mp.Vector3(ag_side_x, side_stack_y, 0)),
    mp.Block(center=mp.Vector3(Al2O3_lside_cx, side_stack_cy, 0),
             material=Al2O3,
             size=mp.Vector3(Al2O3_x, side_stack_y, 0)),
    mp.Block(center=mp.Vector3(Al2O3_rside_cx, side_stack_cy, 0),
             material=Al2O3,
             size=mp.Vector3(Al2O3_x, side_stack_y, 0)),
    mp.Block(center=mp.Vector3(SiO2_lside_cx, side_stack_cy, 0),
             material=SiO2,
             size=mp.Vector3(SiO2_x, side_stack_y, 0)),
    mp.Block(center=mp.Vector3(SiO2_rside_cx, side_stack_cy, 0),
             material=SiO2,
             size=mp.Vector3(SiO2_x, side_stack_y, 0)),
    mp.Block(center=mp.Vector3(0, ag_bottom_cy, 0),
             material=Ag,
             size=mp.Vector3(device_width, ag_bottom_y, 0)),
    mp.Block(center=design_region.center,
             size=design_region.size,
             material=design_variables),
]


def make_oblique_source_left_auto1lambda():
    lam = 1.0 / fcen
    # 타깃 점: 디자인 상단 중심
    y_top = design_cy + design_region_height/2

    # 소스 위치: 디자인 상단에서 정확히 1λ 위 (PML과 충돌하지 않게 살짝 클립)
    y_max_in = Sy/2 - tpml
    y_src = min(y_top + lam, y_max_in - 2/resolution)  # 상부 PML까지 2픽셀 여유

    # x는 좌측에 '붙이기', 길이는 항상 design_region_width
    L_src = design_region_width
    eps   = 1e-6
    x_min_in = -Sx/2
    x_src = x_min_in + L_src/2 + eps

    # 자동 각도 계산: 소스(center) → 타깃(0, y_top) 방향
    dx = 0.0 - x_src        # 타깃 x - 소스 x (좌측이므로 dx>0)
    dy = y_src - y_top      # 항상 >0 (소스가 위)
    theta = np.arctan2(abs(dx), dy)     # [0, π/2]
    k0 = 2*np.pi*fcen*1.0               # 공기 n=1
    kx = np.sign(dx) * k0 * np.sin(theta)

    # 사선 평면파: amp_func + Bloch k_point
    def oblique_amp(r):
        return np.exp(1j * kx * r.x)

    sources = [mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(x_src, y_src, 0),
        size=mp.Vector3(L_src, 0, 0),
        amp_func=oblique_amp
    )]
    k_point = mp.Vector3(kx, 0, 0)
    return sources, k_point

# 사용: 기존 sources/sim 생성부를 아래로 교체
sources, k_point = make_oblique_source_left_auto1lambda()

# 기존 sim 생성부 교체
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    sources=sources,
    boundary_layers=boundary_layers,  # y=PML, x=주기
    k_point=k_point,
    geometry=geometry
)

layer_num = 1
ML = (layer_num > 1)
region_height_each = design_region_height / layer_num
full_center_y = design_cy  

# --- 멀티레이어 디자인 영역 생성/삽입 ---
if ML:
    design_variables = []
    design_region = []
    for dv in range(layer_num):
        mg = mp.MaterialGrid(
            mp.Vector3(Nx, Ny),
            air,     # high index
            SiO2,     # low index
            grid_type="U_MEAN",
            do_averaging=False
        )
        design_variables.append(mg)

        offset = ((layer_num - 1)/2 - dv) * region_height_each
        center_y = full_center_y + offset

        dr = mpa.DesignRegion(
            mg,
            volume=mp.Volume(
                center=mp.Vector3(0, center_y, 0),
                size=mp.Vector3(design_region_width, region_height_each, 0),
            ),
        )
        design_region.append(dr)

        geometry.append(
            mp.Block(
                material=mg,
                size=dr.size,
                center=dr.center
            )
        )
else:
    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny),
        air,   # high index
        SiO2,   # low index
        grid_type="U_MEAN",
        do_averaging=False
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0, full_center_y, 0),
            size=mp.Vector3(design_region_width, region_height_each, 0),
        )
    )
    geometry.append(
        mp.Block(
            material=design_variables,
            size=design_region.size,
            center=design_region.center
        )
    )

dft_fields = mpa.FourierFields(
    sim,
    mp.Volume(center=mp.Vector3(0, y_min_in + ag_bottom_y + gan_y/4, 0), size=mp.Vector3(ag_bottom_x,0,0)),
    mp.Ez,
    yee_grid=True
)

J = lambda fields : npa.sum(npa.abs(fields)**2, axis=1)

opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J],
        objective_arguments=[dft_fields],
        design_regions=[design_region],  
        frequencies=[fcen],
        maximum_run_time=100,
    )

eta_i = 0.5
DESIGN_MODE = 'free' # 또는 free

def mapping_grating(x, eta, beta):
    layer = x.reshape(Nx, Ny)
    proj  = mpa.tanh_projection(layer, beta, eta)
    return proj.ravel()
def mapping_free(x, eta, beta):
    field = x.reshape(Nx, Ny)
    proj  = mpa.tanh_projection(field, beta, eta)
    return proj.ravel()

mapping_fn = mapping_grating if DESIGN_MODE == 'grating' else mapping_free

def update_beta(evaluation_history, binarization_history, beta):
    if len(evaluation_history) >= 3:
        f_prev2 = evaluation_history[-3]
        f_prev1 = evaluation_history[-2]
        f_curr  = evaluation_history[-1]
        bin_prev2 = binarization_history[-3]
        bin_prev1 = binarization_history[-2]
        bin_curr  = binarization_history[-1]
        change1 = abs(np.mean(f_curr)  - np.mean(f_prev1)) / (abs(np.mean(f_prev1)) + 1e-12)
        change2 = abs(np.mean(f_prev1) - np.mean(f_prev2)) / (abs(np.mean(f_prev2)) + 1e-12)
        bin_change1 = abs(bin_curr  - bin_prev1) / (abs(bin_prev1) + 1e-12)
        bin_change2 = abs(bin_prev1 - bin_prev2) / (abs(bin_prev2) + 1e-12)
        if (change1 < 0.005 and change2 < 0.005) and (bin_change1 < 0.001 and bin_change2 < 0.001):
            beta *= 1.2
    return beta


n = Nx * Ny
np.random.seed(4)
x = 0.4 + 0.2 * np.random.rand(n)
# x = np.ones(n)*0.5

cur_beta = 4
cur_iter             = [0]
evaluation_history   = []
beta_history         = []
binarization_history = []

g_histories   = [[] for _ in range(layer_num)]
x_histories   = [[] for _ in range(layer_num)]
lr_histories  = [[] for _ in range(layer_num)]
uf_histories  = [[] for _ in range(layer_num)]
t_histories   = [[] for _ in range(layer_num)]

optimizers = [AdamOptimizer(lr=0.02, beta1=0.9) for _ in range(layer_num)]

def f_multi(v, eta, beta):
    print(f"\n--- Iter {cur_iter[0]+1} ---")

    # β 전환 감지
    beta_changed = (len(beta_history) > 0 and beta != beta_history[-1])
    beta_prev = beta_history[-1] if beta_changed else beta

    # 현재/이전 β에서의 매핑
    rho_full = mapping_fn(v, eta, beta)
    if beta_changed:
        rho_old = mapping_fn(v, eta, beta_prev)

    # 이진화도 기록
    bin_deg = np.sum(np.abs(rho_full - 0.5)) / (0.5 * rho_full.size)
    binarization_history.append(bin_deg)

    size_each = Nx * Ny

    # --- FoM/그래디언트 계산 ---
    f0_list, dJ_list = [], []
    if beta_changed:
        f_new, dJ_new = opt([rho_full], need_value=True, need_gradient=True, beta=beta)
        f_old, dJ_old = opt([rho_old],  need_value=True, need_gradient=True, beta=beta_prev)
        f0_list.append(0.5*(f_new.flatten() + f_old.flatten()))
        dJ_list.append(0.5*(dJ_new + dJ_old))
    else:
        f0_i, dJ_i = opt([rho_full], need_value=True, need_gradient=True, beta=beta)
        f0_list.append(f0_i.flatten()); dJ_list.append(dJ_i)
        
    dJ_total = np.mean(dJ_list, axis=0)
    dJ_flat = np.array(dJ_total).flatten()

    if beta_changed:
        g_new = tensor_jacobian_product(mapping_fn, 0)(v, eta, beta,      dJ_flat)
        g_old = tensor_jacobian_product(mapping_fn, 0)(v, eta, beta_prev, dJ_flat)
        gradient_full = 0.5*(g_new + g_old)
    else:
        gradient_full = tensor_jacobian_product(mapping_fn, 0)(v, eta, beta, dJ_flat)

    g_norm = np.linalg.norm(gradient_full.ravel()) / np.sqrt(gradient_full.size) + 1e-12
    gradient_full = gradient_full / g_norm

    grad_list = [gradient_full[i * size_each:(i + 1) * size_each] for i in range(layer_num)]
    vs = v.reshape(layer_num, size_each)
    v_new_layers = []
    
    for i in range(layer_num):
        vi_new, lr, uf, m, vt, t = optimizers[i].update(vs[i], grad_list[i])
        lr_histories[i].append(lr)
        uf_histories[i].append(uf)
        t_histories[i].append(t)
        g_histories[i].append(grad_list[i].copy())
        x_histories[i].append(vi_new.copy())
        v_new_layers.append(vi_new)

        if not hasattr(optimizers[i], "base_lr"):
            optimizers[i].base_lr = optimizers[i].lr
        if optimizers[i].lr < optimizers[i].base_lr:
            optimizers[i].lr = min(optimizers[i].base_lr, optimizers[i].lr * 1.25)

    v_new = np.concatenate(v_new_layers)

    f_vals = [float(np.abs(f0).item()) for f0 in f0_list]
    evaluation_history.append(f_vals)

    beta_history.append(beta)
    cur_iter[0] += 1

    update_beta(evaluation_history, binarization_history, beta)

    print(f"Current β: {beta:.3f}")
    print("FoM values:", f_vals)
    print(f"Mean FoM: {np.mean(f_vals):.6f}")
    print(f"Binarization degree: {bin_deg:.4f}")

    return v_new, beta

Max_iter = 300

while cur_iter[0] < Max_iter:
    x, cur_beta = f_multi(x, eta_i, cur_beta)
    if binarization_history[-1] > 0.98:
        print("Threshold reached → final mapping with β=∞")
        # x, _ = f_multi(x, eta_i, np.inf)
        break

import matplotlib.pyplot as plt
import os

OUT_DIR = f"resolution{resolution}_{DESIGN_MODE}"

def save_figure(filename, dpi=200):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=dpi, bbox_inches="tight")
    plt.close()

################ Find best FOM ################ 
eval_hist = np.array(evaluation_history)
max_val   = eval_hist.max()
max_idx0  = eval_hist.argmax()
iteration = max_idx0 + 1
print(f"최대 FoM = {max_val:.6f} 는 iteration {iteration} 에서 나왔습니다.")

################ FOM plot ################ 
evaluation_array = np.array(evaluation_history)  
iterations = np.arange(len(evaluation_history))

# [1] 각 소스별 FoM
plt.figure(figsize=(6, 6))
for i in range(evaluation_array.shape[1]):
    plt.plot(iterations, evaluation_array[:, i], label=f"Source {i+1}")
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('FoM (per source)', fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tick_params(direction='in'); plt.grid(True); plt.legend(fontsize=12)
save_figure("fom_per_source.png")

# [2] 평균 FoM
plt.figure(figsize=(6, 6))
mean_fom = evaluation_array.mean(axis=1)
plt.plot(iterations, mean_fom/max(mean_fom), 'b-', label='Mean FoM')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Normalize FoM', fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tick_params(direction='in'); plt.grid(True)
save_figure("fom_mean.png")

# Beta & Binarization
fig, ax1 = plt.subplots(figsize=(6,6))
color1 = 'red'
ax1.set_xlabel('Iteration', fontsize=14)
ax1.set_ylabel('Beta', color=color1, fontsize=14)
ax1.plot(beta_history, color=color1, label='Beta History')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(0, int(cur_iter[0]))
ax1.tick_params(labelsize=14); ax1.grid(True)
ax2 = ax1.twinx()
color2 = 'green'
ax2.set_ylabel('Binarization Degree', color=color2, fontsize=14)
ax2.plot(binarization_history, color=color2, linestyle='-', label='Binarization Degree')
ax2.tick_params(axis='y', labelcolor=color2); ax2.set_ylim(0, 1); ax2.tick_params(labelsize=16)
save_figure("beta_binarization.png")

################ 최종 구조 플롯 저장 ################
rho_final = mapping_fn(x, eta_i, cur_beta)
_ = opt([rho_final], need_value=False, need_gradient=False, beta=cur_beta)
opt.plot2D(True)
save_figure("final_geometry.png")

# 디자인 변수 저장
k = design_variables.weights
np.savetxt(os.path.join(OUT_DIR, "example.txt"), k)
