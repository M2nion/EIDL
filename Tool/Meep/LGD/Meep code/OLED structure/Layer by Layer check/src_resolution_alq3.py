import meep as mp
import numpy as np
import csv

def sourcepower(resolution):

    # === 설정 ===
    cell_size = mp.Vector3(10, 10, 0)
    fcen = 1.75        # Gaussian source 중심 주파수
    df = 3.3           # Gaussian source 대역폭
    nfreq = 300

    field_component = mp.Ex

    # 측정할 파장 범위
    lambda_min = 0.4
    lambda_max = 0.7
    fmin = 1 / lambda_max   # ≈ 1.4286
    fmax = 1 / lambda_min   # = 2.5
    fcen_dft = 0.5 * (fmin + fmax)
    df_dft = fmax - fmin

    # 소스 설정
    source_center = mp.Vector3(0, 0, 0)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            component=field_component,
            center=source_center,
            size=mp.Vector3(0, 0)
        )
    ]

    # PML 설정
    pml_layers = [mp.PML(1)]

    # 시뮬레이션 객체 생성
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        default_material=mp.Medium(index=1.68),
        sources=sources,
        resolution=resolution
    )

    # Flux monitor 추가
    offset = 0.01
    flux_size = 0.02

    flux_monitor = sim.add_flux(
        fcen_dft, df_dft, nfreq,
        mp.FluxRegion(center=source_center - mp.Vector3(offset, 0, 0),
                      size=mp.Vector3(0, flux_size),
                      weight=-1),
        mp.FluxRegion(center=source_center + mp.Vector3(offset, 0, 0),
                      size=mp.Vector3(0, flux_size)),
        mp.FluxRegion(center=source_center - mp.Vector3(0, offset, 0),
                      size=mp.Vector3(flux_size, 0),
                      weight=-1),
        mp.FluxRegion(center=source_center + mp.Vector3(0, offset, 0),
                      size=mp.Vector3(flux_size, 0))
    )

    # 시뮬레이션 실행
    sim.run(until=20)

    # Flux 데이터 얻기
    flux_freqs = np.array(mp.get_flux_freqs(flux_monitor))  # 주파수 배열
    net_power = np.array(mp.get_fluxes(flux_monitor))       # 각 주파수에서의 power

    # 주파수를 파장으로 변환
    wavelengths = 1 / flux_freqs

    # === CSV 저장 ===
    filename = f"sourcepower_res{resolution}.csv"   # 예: sourcepower_res1000.csv

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Wavelength (μm)", "Power (W)"])
        for wl, power in zip(wavelengths, net_power):
            writer.writerow([wl, power])

    print(f"CSV 파일 저장 완료: {filename}")
    return wavelengths, net_power