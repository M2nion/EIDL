import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

# 시뮬레이션 설정
resolution = 50  # pixels/um
sxy = 4  # 시뮬레이션 영역 크기 (단위: um)
dpml = 1  # PML 영역 두께
cell = mp.Vector3(sxy + 2 * dpml, sxy + 2 * dpml)  # 전체 시뮬레이션 영역 크기

pml_layers = [mp.PML(dpml)]  # PML(Perfectly Matched Layer) 설정

fcen = 1.0  # 중심 주파수 (단위: THz)
df = 0.4  # 주파수 대역폭
src_cmpt = mp.Ez  # 원천 전기장 성분 (Ex)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(), component=src_cmpt
    )
]

# 대칭성 설정
if src_cmpt == mp.Ex:
    symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)]
elif src_cmpt == mp.Ey:
    symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1)]
elif src_cmpt == mp.Ez:
    symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]
else:
    symmetries = []

# 시뮬레이션 객체 생성
sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    sources=sources,
    symmetries=symmetries,
    boundary_layers=pml_layers,
)

# Near2Far 및 Flux 설정
nearfield_box = sim.add_near2far(
    fcen,
    0,
    1,
    mp.Near2FarRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(sxy, 0)),
    # mp.Near2FarRegion(
    #     center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(sxy, 0), weight=-1
    # ),
    mp.Near2FarRegion(center=mp.Vector3(+0.5 * sxy, 0), size=mp.Vector3(0, sxy)),
    mp.Near2FarRegion(
        center=mp.Vector3(-0.5 * sxy, 0), size=mp.Vector3(0, sxy), weight=-1
    ),
)

flux_box = sim.add_flux(
    fcen,
    0,
    1,
    mp.FluxRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(sxy, 0)),
    mp.FluxRegion(center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(sxy, 0), weight=-1),
    mp.FluxRegion(center=mp.Vector3(+0.5 * sxy, 0), size=mp.Vector3(0, sxy)),
    mp.FluxRegion(center=mp.Vector3(-0.5 * sxy, 0), size=mp.Vector3(0, sxy), weight=-1),
)

# 시뮬레이션 실행
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

# Far-field 계산
npts = 100  # 각도에 대한 샘플 수
angles = 2 * math.pi / npts * np.arange(npts)

E = np.zeros((npts, 3), dtype=np.complex128)
H = np.zeros((npts, 3), dtype=np.complex128)
for n in range(npts):
    ff = sim.get_farfield(
        nearfield_box, mp.Vector3(r * math.cos(angles[n]), r * math.sin(angles[n]))
    )
    E[n, :] = [ff[j] for j in range(3)]
    H[n, :] = [ff[j + 3] for j in range(3)]

# Poynting 벡터 계산
Px = np.real(np.conj(E[:, 1]) * H[:, 2] - np.conj(E[:, 2]) * H[:, 1])
Py = np.real(np.conj(E[:, 2]) * H[:, 0] - np.conj(E[:, 0]) * H[:, 2])
Pr = np.sqrt(np.square(Px) + np.square(Py))

# integrate the radial flux over the circle circumference
far_flux_circle = np.sum(Pr) * 2 * np.pi * r / len(Pr)

print(f"flux:, {near_flux:.6f}, {far_flux_box:.6f}, {far_flux_circle:.6f}")


# 각도 차이가 0.5도 이하인 경우 추출
angle_threshold = 45  # 각도 차이 (도)
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
fig.savefig(f"filtered_radiation_pattern_{mp.component_name(src_cmpt)}.png", dpi=150, bbox_inches="tight")


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