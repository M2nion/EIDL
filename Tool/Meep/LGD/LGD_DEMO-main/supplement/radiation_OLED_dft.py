import math
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

import meep as mp
import meep.adjoint as mpa

mp.verbosity(0)

# 재료 설정
GLS = mp.Medium(index=1.5)
ITO = mp.Medium(index=1.2)
ORG = mp.Medium(index=1.75)
SiN = mp.Medium(index=1.9)
ALU = mp.Medium(index=3.5)

# 파장 및 주파수 설정
lambda_min = 0.4
lambda_max = 0.8
fmin = 1 / lambda_max
fmax = 1 / lambda_min
fcen = 0.5 * (fmin + fmax)
df = fmax - fmin

# 시뮬레이션 파라미터
resolution = 50
nfreq = 25

# 두께 설정
tABS = 0.5
tPML = 0.5
tGLS = 0.5
tITO = 0.5
tORG = 0.5
tALU = 0.2

L = 1.0
H = tGLS + 0.2 + tITO + tORG + tALU

Sy = tPML + tGLS + 0.2 + tITO + tORG + tALU
Sx = L + 2 * tABS
cell_size = mp.Vector3(Sx, Sy)

boundary_layers = [
    mp.Absorber(tABS, direction=mp.X),
    mp.PML(tPML, direction=mp.Y, side=mp.High),
]

# 디자인 영역
Nx, Ny = 50, 50
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), GLS, SiN, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(y=0.5 * Sy - tPML - tGLS - 0.5 * 0.2),
        size=mp.Vector3(L, 0.2, 0),
    ),
)

# 기하 구조
geometry = [
    mp.Block(
        material=GLS,
        size=mp.Vector3(mp.inf, tPML + tGLS),
        center=mp.Vector3(y=0.5 * Sy - 0.5 * (tPML + tGLS)),
    ),
    mp.Block(
        material=design_variables,
        size=design_region.size,
        center=design_region.center,
    ),
    mp.Block(
        material=ITO,
        size=mp.Vector3(mp.inf, tITO),
        center=mp.Vector3(y=0.5 * Sy - tPML - tGLS - 0.5 * tITO),
    ),
    mp.Block(
        material=ORG,
        size=mp.Vector3(mp.inf, tORG),
        center=mp.Vector3(y=0.5 * Sy - tPML - tGLS - tITO - 0.5 * tORG),
    ),
    mp.Block(
        material=ALU,
        size=mp.Vector3(mp.inf, tALU),
        center=mp.Vector3(y=0.5 * Sy - tPML - tGLS - tITO - tORG - 0.5 * tALU),
    ),
]

# 광원 설정
src_cmpt = mp.Ez
sources = [
    mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
        component=src_cmpt,
        center=mp.Vector3(y=0.5 * Sy - tPML - tGLS - 0.5 * 0.2),
    )
]

# 시뮬레이션 실행
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    geometry=geometry,
    sources=sources,
)

# 필드 추출 설정
sim.run(
    mp.at_end(mp.output_efield_z),
    mp.at_end(mp.output_hfield_x),
    mp.at_end(mp.output_hfield_y),
    until=200,
)
# ... (위 코드 모두 동일, sim.run()까지 동일)

# 필드 배열 추출
ez = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
hx = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Hx)
hy = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Hy)

# 격자 크기
Nx, Ny = ez.shape

# 사방 경계에서 Poynting 벡터 계산
# 위쪽 (top)
top_y = Ny - int(tPML * resolution) - 1
S_top_x = np.real(np.conj(ez[:, top_y]) * hy[:, top_y])
S_top_y = -np.real(np.conj(ez[:, top_y]) * hx[:, top_y])

# 아래쪽 (bottom)
bot_y = int(tPML * resolution)
S_bot_x = np.real(np.conj(ez[:, bot_y]) * hy[:, bot_y])
S_bot_y = -np.real(np.conj(ez[:, bot_y]) * hx[:, bot_y])

# 오른쪽 (right)
right_x = Nx - int(tABS * resolution) - 1
S_right_x = np.real(np.conj(ez[right_x, :]) * hy[right_x, :])
S_right_y = -np.real(np.conj(ez[right_x, :]) * hx[right_x, :])

# 왼쪽 (left)
left_x = int(tABS * resolution)
S_left_x = np.real(np.conj(ez[left_x, :]) * hy[left_x, :])
S_left_y = -np.real(np.conj(ez[left_x, :]) * hx[left_x, :])

# 각도 계산 (0 ~ 2pi)
angles = []
powers = []

# 위쪽 (angle: -90° ~ 90°)
for i in range(Nx):
    theta = math.atan2(+1, (i - Nx / 2) / (Nx / 2))
    angles.append(theta)
    powers.append(np.sqrt(S_top_x[i] ** 2 + S_top_y[i] ** 2))

# 오른쪽 (90° ~ 270°)
for j in range(Ny):
    theta = math.atan2((Ny / 2 - j) / (Ny / 2), +1)
    angles.append(theta)
    powers.append(np.sqrt(S_right_x[j] ** 2 + S_right_y[j] ** 2))

# 아래쪽 (270° ~ 450° → 정규화 필요)
for i in range(Nx):
    theta = math.atan2(-1, (Nx / 2 - i) / (Nx / 2))
    angles.append(theta)
    powers.append(np.sqrt(S_bot_x[i] ** 2 + S_bot_y[i] ** 2))

# 왼쪽 (180° ~ 360° → 정규화 필요)
for j in range(Ny):
    theta = math.atan2((j - Ny / 2) / (Ny / 2), -1)
    angles.append(theta)
    powers.append(np.sqrt(S_left_x[j] ** 2 + S_left_y[j] ** 2))

# 정렬
angles = np.array(angles)
powers = np.array(powers)
sort_idx = np.argsort(angles)
angles_sorted = angles[sort_idx]
powers_sorted = powers[sort_idx]

# 정규화
powers_sorted /= np.max(powers_sorted)

# 플로팅
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
ax.plot(angles_sorted, powers_sorted, label="Flux (4-side)", color="blue")
ax.set_rmax(1)
ax.set_rticks([0.5, 1])
ax.set_rlabel_position(22)
ax.grid(True)
ax.legend()
fig.savefig("OLED_flux_pattern_all_sides.png", dpi=150, bbox_inches="tight")
