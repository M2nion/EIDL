import os
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import math

from meep.materials import Ag

mp.verbosity(0)

fontlabel = 16

lambda_min = 0.4       # 최소 파장 (µm)
lambda_max = 0.7       # 최대 파장 (µm)
fmin = 1/lambda_max    # 최소 주파수
fmax = 1/lambda_min    # 최대 주파수
fcen = 0.5*(fmin+fmax) # 중앙 주파수

resolution = 50        # 시뮬레이션 해상도
design_region_resolution = 50

# nfreq = 50             # 추출할 주파수 개수
df = fmax-fmin         # 주파수 대역폭
nfreq = 1

tag_bot = 0.5
tag_side = 4
tsio2 = 4
monitor_position = 0.5 * 2
monitor_blank = 0.5 * 0.1
tpml = 0.4

width_ag = 0.2
width_sio2 = 2
width_bot_ag = round(width_ag + width_sio2 + width_ag,2)
design_region_width = round(width_bot_ag, 3)
design_region_height = 0.5

# 시뮬레이션 셀 크기
Sx = tpml + width_bot_ag + tpml
Sy = round(tpml + tag_bot + tag_side + design_region_height + monitor_position + monitor_blank + tpml, 3)
cell_size = mp.Vector3(Sx, Sy)

# 재료
SiO2 = mp.Medium(index=1.45)
TiO2 = mp.Medium(index=2.6)
Air = mp.Medium(index=1.0)

# === Design mode 선택: 'grating' 또는 'free'
DESIGN_MODE = 'grating'   # 'free' 로 바꾸면 freeform 사용
DESIGN_MODE = 'free'

# 디자인 변수 격자 크기
Nx = int(design_region_resolution * design_region_width) + 1
if DESIGN_MODE == 'grating':
    Ny = 1
else:
    Ny = int(design_region_resolution * design_region_height) + 1

# 위치 계산
y_bottom = -Sy / 2
center_y_ag = y_bottom + tpml + tag_bot / 2
center_y_sio2 = y_bottom + tpml + tag_bot + tsio2 / 2
center_y_design = y_bottom + tpml + tag_bot + tsio2 + design_region_height / 2
center_y_monitor_position = y_bottom + tpml + tag_bot + tsio2 + design_region_height + monitor_position
center_y_source_position = y_bottom + tpml + tag_bot + 1 / fcen
center_Ag_x_position = design_region_width / 2 - width_ag / 2

boundary_layers = [
    mp.Absorber(tpml, direction=mp.X),
    mp.PML(tpml, direction=mp.Y)
]

# 기본 지오메트리 정의
geometry = [
    # Bottom Ag layer
    mp.Block(
        material=Ag,
        size=mp.Vector3(width_bot_ag, tag_bot, 0),
        center=mp.Vector3(0, center_y_ag, 0)
    ),
    # SiO2 layer
    mp.Block(
        material=SiO2,
        size=mp.Vector3(width_sio2, tsio2, 0),
        center=mp.Vector3(0, center_y_sio2, 0)
    ),
    # Side metal
    mp.Block(
        material=Ag,
        size=mp.Vector3(width_ag, tsio2, 0),
        center=mp.Vector3(-center_Ag_x_position, center_y_sio2, 0)
    ),
    mp.Block(
        material=Ag,
        size=mp.Vector3(width_ag, tsio2, 0),
        center=mp.Vector3(center_Ag_x_position, center_y_sio2, 0)
    ),
]

# 레이어 수
layer_num = 1
ML = (layer_num > 1)
region_height_each = design_region_height / layer_num
full_center_y = y_bottom + tag_bot + tsio2 + design_region_height/2

if ML:
    design_variables = []
    design_region = []
    for dv in range(layer_num):
        mg = mp.MaterialGrid(
            mp.Vector3(Nx, Ny),
            TiO2,       
            SiO2,       
            grid_type="U_MEAN",
            do_averaging=False
        )
        design_variables.append(mg)
        offset = ((layer_num - 1) / 2 - dv) * region_height_each
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
                material=design_variables[dv],
                size=design_region[dv].size,
                center=design_region[dv].center
            )
        )
else:
    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, Ny),
        TiO2,
        SiO2,
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

num_sources = 1
spacing = 10 / resolution
offsets = [ (i - (num_sources - 1) // 2) * spacing for i in range(num_sources) ]
source_x_position = 0.5

sources = [
    mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=mp.Vector3(source_x_position, center_y_monitor_position, 0)
    )
    for offset in offsets
]

# ====== 출력 폴더 설정 ======
ndipole = num_sources
OUT_DIR = f"ndipole{ndipole}_resolution{resolution}_{DESIGN_MODE}_loc{source_x_position}"
os.makedirs(OUT_DIR, exist_ok=True)

def save_figure(filename, dpi=200):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=dpi, bbox_inches="tight")
    plt.close()

k0 = mp.Vector3(0,0,0)

sims = [
    mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=[src], 
        default_material=mp.Medium(index=1),
        extra_materials=[Ag],
        # k_point=k0
    )
    for src in sources
]

monitor_position_J = mp.Vector3(0, center_y_source_position)
monitor_size = mp.Vector3(width_sio2-2/resolution, 0)

FourierFields_list = [
    mpa.FourierFields(
        sims[i],  
        mp.Volume(center=monitor_position_J, size=monitor_size),
        mp.Ez,
        yee_grid=True
    )
    for i in range(num_sources)
]

J = lambda fields : npa.sum(npa.abs(fields)**2, axis=1)

opt_list = [
    mpa.OptimizationProblem(
        simulation=sims[i],
        objective_functions=[J],
        objective_arguments=[FourierFields_list[i]],
        design_regions=[design_region],  
        frequencies=[fcen],
        maximum_run_time=100,
    )
    for i in range(num_sources)
]

# 초기 구조 플롯 저장
for i in range(num_sources):
    opt_list[i].plot2D(True)
    save_figure(f"init_geometry_src{i+1}.png")

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
    
eta_i = 0.5

################ Nx * 1 grating design ################ 
def grating_mapping(x, eta, beta):    
    size_each = Nx * Ny
    v3 = x.reshape(layer_num, size_each)
    rho_list = []
    for i in range(layer_num):
        layer_field = v3[i].reshape(Nx, Ny)
        # sym_field   = (layer_field[::-1, :] + layer_field) / 2
        sym_field   = layer_field
        flat        = sym_field.ravel()
        proj        = mpa.tanh_projection(flat, beta, eta)
        rho_list.append(proj)
    return npa.concatenate(rho_list, axis=0)

################ Nx * Ny freeform design ################ 
def free_mapping(x, eta, beta):
    size_each = Nx * Ny
    v3 = x.reshape(layer_num, size_each).reshape(layer_num, Nx, Ny)
    rho_list = []
    for i in range(layer_num):
        field = v3[i]
        proj  = mpa.tanh_projection(field, beta, eta)
        # proj = 0.5*(proj[::-1, :] + proj)   # x-축(좌우) 대칭
        rho_list.append(proj.ravel())
    return npa.concatenate(rho_list, axis=0)

# 매핑 함수 선택자
mapping_fn = grating_mapping if DESIGN_MODE == 'grating' else free_mapping
print(f"[Design Mode] {DESIGN_MODE} (Nx={Nx}, Ny={Ny})")

############ Beta update ############
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
    for i in range(len(opt_list)):
        if beta_changed:
            f_new, dJ_new = opt_list[i]([rho_full], need_value=True, need_gradient=True, beta=beta)
            f_old, dJ_old = opt_list[i]([rho_old],  need_value=True, need_gradient=True, beta=beta_prev)
            f0_list.append(0.5*(f_new.flatten() + f_old.flatten()))
            dJ_list.append(0.5*(dJ_new + dJ_old))
        else:
            f0_i, dJ_i = opt_list[i]([rho_full], need_value=True, need_gradient=True, beta=beta)
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

mp.verbosity(0)
Max_iter = 300

while cur_iter[0] < Max_iter:
    x, cur_beta = f_multi(x, eta_i, cur_beta)
    if binarization_history[-1] > 0.98:
        print("Threshold reached → final mapping with β=∞")
        # x, _ = f_multi(x, eta_i, np.inf)
        break

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
_ = opt_list[0]([rho_final], need_value=False, need_gradient=False, beta=cur_beta)
opt_list[0].plot2D(True)
save_figure("final_geometry.png")

# 디자인 변수 저장
k = design_variables.weights
np.savetxt(os.path.join(OUT_DIR, "example.txt"), k)
