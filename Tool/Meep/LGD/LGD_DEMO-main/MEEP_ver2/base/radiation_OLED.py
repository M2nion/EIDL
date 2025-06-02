import math
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

import meep as mp
import meep.adjoint as mpa
import numpy as np
from matplotlib import pyplot as plt

from meep.materials import Al as ALU

mp.verbosity(0)

# 재료 설정
GLS = mp.Medium(index=1.5)    # 글래스
ITO = mp.Medium(index=1.2)    # ITO
ORG = mp.Medium(index=1.75)   # OLED 유기층
SiN = mp.Medium(index=1.9)    # 실리콘 질화물

# 파장 및 주파수 범위 설정
lambda_min = 0.4       # 최소 파장 (µm)
lambda_max = 0.8       # 최대 파장 (µm)
fmin = 1/lambda_max    # 최소 주파수
fmax = 1/lambda_min    # 최대 주파수
fcen = 0.5*(fmin+fmax) # 중앙 주파수
df = fmax-fmin         # 주파수 대역폭

# 해상도 및 설계 영역 설정
resolution = 50
nfreq = 25  # 추출할 주파수 개수
design_region_resolution =50

# 두께 설정
tABS = 0.5    # X/Y 방향 흡수 경계층 두께
tPML = 0.5    # Y 방향 PML 경계층 두께
tGLS = 0.5    # 글래스 층 두께
tITO = 0.5    # ITO 층 두께
tORG = 0.5    # 유기층(OLED 발광층) 두께
tALU = 0.2    # 알루미늄(캐소드) 두께
design_region_height = 0.2 # 디자인 영역 높이

L = 1.0  # OLED의 가로/세로 길이
H = tGLS + design_region_height + tITO + tORG + tALU  # OLED의 세로 길이

design_region_width = L

# 시뮬레이션 셀 크기
Sy = tPML + tGLS + design_region_height + tITO + tORG + tALU  # Z 방향 전체 길이
Sx = L + 2*tABS  # X/Y 방향 길이
cell_size = mp.Vector3(Sx, Sy)

# 경계층
boundary_layers = [mp.Absorber(tABS,direction=mp.X),
                   mp.PML(tPML,direction=mp.Y,side=mp.High)]

# 디자인 영역
Nx = int(design_region_resolution * design_region_width) + 1
Ny = int(design_region_resolution * design_region_height) + 1
structure_weights = np.loadtxt(f'lastdesign.txt')
design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), GLS, SiN, grid_type="U_MEAN")
design_variables.update_weights(structure_weights)
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(y=0.5*Sy - tPML - tGLS - 0.5*design_region_height),
        size=mp.Vector3(L, design_region_height, 0),
    ),
)

# 기하학적 구조 정의
geometry = [
    mp.Block(material=GLS,      # 유리층
             size=mp.Vector3(mp.inf, tPML + tGLS),
             center=mp.Vector3(y=0.5*Sy - 0.5*(tPML + tGLS))),

    mp.Block(material=design_variables,      # 디자인 영역
             size=design_region.size,
             center=design_region.center),
    
    mp.Block(material=ITO,      # ITO 층
             size=mp.Vector3(mp.inf, tITO),
             center=mp.Vector3(y=0.5*Sy - tPML - tGLS - design_region_height - 0.5*tITO)),
    
    mp.Block(material=ORG,                       # 유기층
             size=mp.Vector3(mp.inf, tORG),
             center=mp.Vector3(y=0.5*Sy - tPML - tGLS - design_region_height - tITO - 0.5*tORG)),
    
    mp.Block(material=ALU,                       # 알루미늄 층
             size=mp.Vector3(mp.inf, tALU),
             center=mp.Vector3(y=0.5*Sy - tPML - tGLS - design_region_height - tITO - tORG - 0.5*tALU))
]

# 광원 설정
src_cmpt = mp.Ez
sources = [
    mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
              component=src_cmpt,
              center=mp.Vector3(y=0.5*Sy - tPML - tGLS - design_region_height - tITO - 0.5*tORG))
]

# 시뮬레이션 설정
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry,
                    sources=sources,)

nearfield_box = sim.add_near2far(
    fcen,
    0,
    1,
    mp.Near2FarRegion(center=mp.Vector3(0, -tPML*0.5 +0.5 * H), size=mp.Vector3(L, 0)),
    # mp.Near2FarRegion(
    #     center=mp.Vector3(0, -tPML*0.5 -0.5 * H), size=mp.Vector3(L, 0), weight=-1
    # ),
    # mp.Near2FarRegion(center=mp.Vector3(+0.5 * L, 0), size=mp.Vector3(0, H)),
    # mp.Near2FarRegion(
    #     center=mp.Vector3(-0.5 * L, 0), size=mp.Vector3(0, H), weight=-1
    # ),
)

flux_box = sim.add_flux(
    fcen,
    0,
    1,
    mp.FluxRegion(center=mp.Vector3(0, -tPML*0.5 +0.5 * H), size=mp.Vector3(L, 0)),
    mp.FluxRegion(center=mp.Vector3(0, -tPML*0.5 -0.5 * H), size=mp.Vector3(L, 0), weight=-1),
    mp.FluxRegion(center=mp.Vector3(+0.5 * L, -tPML*0.5), size=mp.Vector3(0, H)),
    mp.FluxRegion(center=mp.Vector3(-0.5 * L, -tPML*0.5), size=mp.Vector3(0, H), weight=-1),
)

f = plt.figure(dpi=150)
sim.plot2D(ax=f.gca())
plt.savefig("figure.png")

sim.run(until_after_sources=mp.stop_when_dft_decayed())

near_flux = mp.get_fluxes(flux_box)[0]

# half side length of far-field square box OR radius of far-field circle
r = 1000 / fcen

# resolution of far fields (points/μm)
res_ff = 1

far_flux_box = (
    nearfield_box.flux(
        mp.Y, mp.Volume(center=mp.Vector3(y=r), size=mp.Vector3(2 * r)), res_ff
    )[0]
    - nearfield_box.flux(
        mp.Y, mp.Volume(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r)), res_ff
    )[0]
    + nearfield_box.flux(
        mp.X, mp.Volume(center=mp.Vector3(r), size=mp.Vector3(y=2 * r)), res_ff
    )[0]
    - nearfield_box.flux(
        mp.X, mp.Volume(center=mp.Vector3(-r), size=mp.Vector3(y=2 * r)), res_ff
    )[0]
)

npts = 100  # number of points in [0,2*pi) range of angles
angles = 2 * math.pi / npts * np.arange(npts)

E = np.zeros((npts, 3), dtype=np.complex128)
H = np.zeros((npts, 3), dtype=np.complex128)
for n in range(npts):
    ff = sim.get_farfield(
        nearfield_box, mp.Vector3(r * math.cos(angles[n]), r * math.sin(angles[n]))
    )
    E[n, :] = [ff[j] for j in range(3)]
    H[n, :] = [ff[j + 3] for j in range(3)]

Px = np.real(np.conj(E[:, 1]) * H[:, 2] - np.conj(E[:, 2]) * H[:, 1])
Py = np.real(np.conj(E[:, 2]) * H[:, 0] - np.conj(E[:, 0]) * H[:, 2])
Pr = np.sqrt(np.square(Px) + np.square(Py))

# integrate the radial flux over the circle circumference
far_flux_circle = np.sum(Pr) * 2 * np.pi * r / len(Pr)

print(f"flux:, {near_flux:.6f}, {far_flux_box:.6f}, {far_flux_circle:.6f}")

# Analytic formulas for the radiation pattern as the Poynting vector
# of an electric dipole in vacuum. From Section 4.2 "Infinitesimal Dipole"
# of Antenna Theory: Analysis and Design, 4th Edition (2016) by C. Balanis.
if src_cmpt == mp.Ex:
    flux_theory = np.sin(angles) ** 2
elif src_cmpt == mp.Ey:
    flux_theory = np.cos(angles) ** 2
elif src_cmpt == mp.Ez:
    flux_theory = np.ones((npts,))

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
ax.plot(angles, Pr / max(Pr), "b-", label="Meep")
ax.plot(angles, flux_theory, "r--", label="theory")
ax.set_rmax(1)
ax.set_rticks([0, 0.5, 1])
ax.grid(True)
ax.set_rlabel_position(22)
ax.legend()

if mp.am_master():
    fig.savefig(
        f"OLED_radiation_pattern_{mp.component_name(src_cmpt)}.png",
        dpi=150,
        bbox_inches="tight",
    )

    
# 각도 차이가 0.5도 이하인 경우 추출
angle_threshold = 0.5  # 각도 차이 (도)
angle_threshold_rad = np.radians(angle_threshold)  # 각도 차이를 라디안으로 변환

# +y 방향은 π/2
y_direction = np.pi / 2

# 각도 차이 계산
angle_diffs = np.abs(angles - y_direction)

# 0.5도 이하인 인덱스 추출
valid_indices = np.where(angle_diffs <= angle_threshold_rad)[0]

# 유효한 각도에서의 E, H 값 추출
E_valid = E[valid_indices, :]
H_valid = H[valid_indices, :]

# Poynting 벡터 계산 (유효한 각도만)
Px_valid = np.real(np.conj(E_valid[:, 1]) * H_valid[:, 2] - np.conj(E_valid[:, 2]) * H_valid[:, 1])
Py_valid = np.real(np.conj(E_valid[:, 2]) * H_valid[:, 0] - np.conj(E_valid[:, 0]) * H_valid[:, 2])
Pr_valid = np.sqrt(np.square(Px_valid) + np.square(Py_valid))

# 시각화
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
ax.plot(angles[valid_indices], Pr_valid / max(Pr_valid), 'bo', label="Filtered Meep")  # 유효한 값만 plot
ax.set_rmax(1)
ax.set_rticks([0, 0.5, 1])
ax.grid(True)
ax.set_rlabel_position(22)
ax.set_title("Radiation Pattern for +y Direction (Filtered)", va='bottom')
ax.legend()

# 결과 이미지 저장
fig.savefig(f"filtered_OLED_radiation_pattern_{mp.component_name(src_cmpt)}.png", dpi=150, bbox_inches="tight")


# 결과 출력
print(f"유효한 각도에서의 Poynting 벡터 (Pr):", np.sum(Pr_valid) * 2 * np.pi * r / len(Pr))

# 전체 Poynting 벡터 크기의 합 (전체 에너지)
total_power = np.sum(Pr)

# 유효 각도에서의 Poynting 벡터 크기의 합 (해당 방향 에너지)
valid_power = np.sum(Pr_valid)

# 에너지 비율 계산 (%)
valid_power_percentage = (valid_power / total_power) * 100

# 출력
print(f"+y 방향 (±0.5도 이내) 방사 에너지 비율: {valid_power_percentage:.2f}%")