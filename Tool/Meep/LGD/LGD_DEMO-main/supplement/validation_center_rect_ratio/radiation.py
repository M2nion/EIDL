import math
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import meep as mp

# 공통 설정
resolution = 50
sxy = 4
dpml = 1
fcen = 1.0
df = 0.4
src_cmpt = mp.Ez
src = mp.GaussianSource(fcen, fwidth=df)
r = 1000 / fcen
res_ff = 1
npts = 100
angles = 2 * math.pi / npts * np.arange(npts)
y_direction = np.pi / 2
angle_threshold_deg = 60
angle_threshold_rad = np.radians(angle_threshold_deg)
angle_diffs = np.abs(angles - y_direction)
valid_indices = np.where(angle_diffs <= angle_threshold_rad)[0]

# 반복할 ratio 값들
ratio_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for ratio in ratio_list:
    
    # 폴더 생성
    folder = f"results_ratio_{ratio}"
    os.makedirs(folder, exist_ok=True)

    cell = mp.Vector3(ratio * sxy + 2 * dpml, sxy + 2 * dpml)
    pml_layers = [mp.PML(dpml)]
    sources = [mp.Source(src=src, center=mp.Vector3(), component=src_cmpt)]

    if src_cmpt == mp.Ex:
        symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)]
    elif src_cmpt == mp.Ey:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1)]
    elif src_cmpt == mp.Ez:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]
    else:
        symmetries = []

    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        sources=sources,
        symmetries=symmetries,
        boundary_layers=pml_layers,
    )

    nearfield_box = sim.add_near2far(
        fcen, 0, 1,
        mp.Near2FarRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(ratio * sxy, 0)),
        # mp.Near2FarRegion(
        #     center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(ratio*sxy, 0), weight=-1
        # ),
        # mp.Near2FarRegion(center=mp.Vector3(+(ratio/2) * sxy, 0), size=mp.Vector3(0, sxy)),
        # mp.Near2FarRegion(
        #     center=mp.Vector3(-(ratio/2) * sxy, 0), size=mp.Vector3(0, sxy), weight=-1
        # ),
    )

    flux_box = sim.add_flux(
        fcen, 0, 1,
        mp.FluxRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(ratio * sxy, 0)),
        # mp.FluxRegion(center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(ratio*sxy, 0), weight=-1),
        # mp.FluxRegion(center=mp.Vector3(+(ratio/2) * sxy, 0), size=mp.Vector3(0, sxy)),
        # mp.FluxRegion(center=mp.Vector3(-(ratio/2) * sxy, 0), size=mp.Vector3(0, sxy), weight=-1),

    )

    sim.plot2D()
    if mp.am_master():
        plt.savefig(os.path.join(folder, "2Dplot.png"), dpi=150, bbox_inches="tight")
        plt.close()

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    near_flux = mp.get_fluxes(flux_box)[0]

    far_flux_box = (
        nearfield_box.flux(mp.Y, mp.Volume(center=mp.Vector3(y=r), size=mp.Vector3(2 * r)), res_ff)[0]
        - nearfield_box.flux(mp.Y, mp.Volume(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r)), res_ff)[0]
        + nearfield_box.flux(mp.X, mp.Volume(center=mp.Vector3(r), size=mp.Vector3(y=2 * r)), res_ff)[0]
        - nearfield_box.flux(mp.X, mp.Volume(center=mp.Vector3(-r), size=mp.Vector3(y=2 * r)), res_ff)[0]
    )

    E = np.zeros((npts, 3), dtype=np.complex128)
    H = np.zeros((npts, 3), dtype=np.complex128)
    for n in range(npts):
        ff = sim.get_farfield(nearfield_box, mp.Vector3(r * math.cos(angles[n]), r * math.sin(angles[n])))
        E[n, :] = [ff[j] for j in range(3)]
        H[n, :] = [ff[j + 3] for j in range(3)]

    Px = np.real(np.conj(E[:, 1]) * H[:, 2] - np.conj(E[:, 2]) * H[:, 1])
    Py = np.real(np.conj(E[:, 2]) * H[:, 0] - np.conj(E[:, 0]) * H[:, 2])
    Pr = np.sqrt(np.square(Px) + np.square(Py))
    far_flux_circle = np.sum(Pr) * 2 * np.pi * r / len(Pr)

    # 이론적 방사 패턴
    if src_cmpt == mp.Ex:
        flux_theory = np.sin(angles) ** 2
    elif src_cmpt == mp.Ey:
        flux_theory = np.cos(angles) ** 2
    elif src_cmpt == mp.Ez:
        flux_theory = np.ones((npts,))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(angles, Pr / max(Pr), "b-", label="Meep")
    ax.plot(angles, flux_theory, "r--", label="Theory")
    ax.set_rmax(1)
    ax.set_rticks([0, 0.5, 1])
    ax.grid(True)
    ax.set_rlabel_position(22)
    ax.legend()
    if mp.am_master():
        fig.savefig(os.path.join(folder, f"radiation_pattern_{mp.component_name(src_cmpt)}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 필터된 각도 분석
    E_valid = E[valid_indices, :]
    H_valid = H[valid_indices, :]
    Px_valid = np.real(np.conj(E_valid[:, 1]) * H_valid[:, 2] - np.conj(E_valid[:, 2]) * H_valid[:, 1])
    Py_valid = np.real(np.conj(E_valid[:, 2]) * H_valid[:, 0] - np.conj(E_valid[:, 0]) * H_valid[:, 2])
    Pr_valid = np.sqrt(np.square(Px_valid) + np.square(Py_valid))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(angles[valid_indices], Pr_valid / max(Pr_valid), 'bo', label="Filtered Meep")
    ax.set_rmax(1)
    ax.set_rticks([0, 0.5, 1])
    ax.grid(True)
    ax.set_rlabel_position(22)
    ax.set_title("+y (±60)")
    ax.legend()
    if mp.am_master():
        fig.savefig(os.path.join(folder, f"filtered_radiation_pattern_{mp.component_name(src_cmpt)}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 요약 출력 및 저장
    total_power = np.sum(Pr)
    valid_power = np.sum(Pr_valid)
    valid_power_percentage = (valid_power / total_power) * 100

    # 필터 각도 범위 내 편차 기반 오차율 계산
    Pr_valid_normalized = Pr_valid / np.max(Pr)  # 전체에서 정규화
    mean_valid = np.mean(Pr_valid_normalized)
    error_relative = np.abs(Pr_valid_normalized - mean_valid) / mean_valid
    uniformity_error_percentage = np.mean(error_relative) * 100

    summary_text = (
        f"ratio = {ratio}\n"
        f"near_flux: {near_flux:.6f}\n"
        f"far_flux_box: {far_flux_box:.6f}\n"
        f"far_flux_circle: {far_flux_circle:.6f}\n"
        f"+y 방향 (±{angle_threshold_deg}도 이내) 방사 에너지 비율: {valid_power_percentage:.2f}%\n"
        f"방사 패턴 균일도 기반 오차율 (필터된 각도 내 편차): {uniformity_error_percentage:.2f}%\n"
    )

    print(summary_text)

    with open(os.path.join(folder, "summary.txt"), "w") as f:
        f.write(summary_text)