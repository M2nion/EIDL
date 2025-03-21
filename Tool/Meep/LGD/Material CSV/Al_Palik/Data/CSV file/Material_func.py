"""
사용법은 예시 코드(https://github.com/NanoComp/meep/blob/master/python/examples/eps_fit_lorentzian.py) 
코드의 하단부 if __name__ =="__main__": 아래부터 제 코드 붙여넣으신 후 사용하시면 됩니다.
"""

import meep as mp
import os
import matplotlib.pyplot as plt
import numpy as np

def Material_fit(Material_data_csv="Al_palik_data.csv", 
                   FDTD_material_csv="Al_palik_data_FDTD.csv",
                   eps_inf=1,
                   fit_wl_min=0.38, fit_wl_max=0.73,
                   num_lorentzians=2, iteration=50,
                   save_path="."):    
    # 저장 경로가 없으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. 데이터 로드 및 전처리
    data = np.genfromtxt(Material_data_csv, delimiter=",")
    wl = data[:, 0] * 1e6  # 파장 (µm)
    n_data = data[:, 1] + 1j * data[:, 2]
    eps_all = np.square(n_data) - eps_inf

    # 피팅 범위로 데이터 선택
    mask = (wl >= fit_wl_min) & (wl <= fit_wl_max)
    wl_fit, eps_fit = wl[mask], eps_all[mask]
    freqs_fit = 1 / wl_fit

    # 2. Lorentzian 피팅 최적화
    ps = np.zeros((iteration, 3 * num_lorentzians))
    errors = np.zeros(iteration)
    for m in range(iteration):
        p_rand = [10 ** np.random.random() for _ in range(3 * num_lorentzians)]
        ps[m, :], errors[m] = lorentzfit(p_rand, freqs_fit, eps_fit, nlopt.LD_MMA, 1e-25, 50000)
        print(f"Iteration {m:3d}, error: {errors[m]:.6f}")
    best = np.argmin(errors)
    print(f"Optimal error: {errors[best]:.6f}")

    # 3. 최적 파라미터로 Susceptibility 생성 및 Meep 모델 구성
    suscept = []
    for i in range(num_lorentzians):
        freq_param = ps[best][3*i + 1]
        gamma = ps[best][3*i + 2]
        if freq_param == 0:
            sigma = ps[best][3*i + 0]
            suscept.append(mp.DrudeSusceptibility(frequency=1.0, gamma=gamma, sigma=sigma))
        else:
            sigma = ps[best][3*i + 0] / freq_param**2
            suscept.append(mp.LorentzianSusceptibility(frequency=freq_param, gamma=gamma, sigma=sigma))
    material = mp.Medium(epsilon=eps_inf, E_susceptibilities=suscept)
    model_eps = [material.epsilon(f)[0][0] for f in freqs_fit]

    # 4. Material 데이터와 Meep 모델 비교 플롯 생성
    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    
    # Re(ε) 플롯
    ax[0].plot(wl_fit, np.real(eps_fit) + eps_inf, 'gs', markersize=4, label="Material data")
    ax[0].plot(wl_fit, np.real(model_eps), 'b-', label="Meep model")
    ax[0].set_xlabel("wavelength (µm)")
    ax[0].set_ylabel(r"Re($\epsilon$)")
    ax[0].grid(True)
    ax[0].set_xlim([fit_wl_min+0.02, fit_wl_max-0.03])
    
    # Im(ε) 플롯
    ax[1].plot(wl_fit, np.imag(eps_fit), 'gs', markersize=4, label="Material data")
    ax[1].plot(wl_fit, np.imag(model_eps), 'b-', label="Meep model")
    ax[1].set_xlabel("wavelength (µm)")
    ax[1].set_ylabel(r"Im($\epsilon$)")
    ax[1].grid(True)
    ax[1].set_xlim([fit_wl_min+0.02, fit_wl_max-0.03])
    
    fig.suptitle("Comparison of Material Data and FDTD Model\n(using Drude-Lorentzian Susceptibility)", fontsize=9)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(save_path, "eps_fit_sample.png"), dpi=150, bbox_inches="tight")

    # 5. FDTD 비교 데이터 로드 및 플롯 추가
    csv = np.genfromtxt(FDTD_material_csv, delimiter=",", skip_header=1)
    csv_wl = csv[:, 0]
    csv_n = csv[:, 1] + 1j * csv[:, 2]
    csv_eps = np.square(csv_n) - eps_inf

    ax[0].plot(csv_wl, np.real(csv_eps) + eps_inf, 'r--', markersize=6, label="FDTD model")
    ax[1].plot(csv_wl, np.imag(csv_eps), 'r--', markersize=6, label="FDTD model")
    ax[0].legend()
    ax[1].legend()
    fig.savefig(os.path.join(save_path, "eps_fit_sample_with_csv.png"), dpi=150, bbox_inches="tight")
    print(f"FDTD 데이터가 추가된 플롯을 지정된 경로로 저장했습니다.")

    # 6. 결과 CSV 파일 저장
    fdtd_save = os.path.join(save_path, "fdtd_fit_data.csv")
    np.savetxt(fdtd_save, np.column_stack((csv_wl, np.real(csv_eps) + eps_inf, np.imag(csv_eps))), delimiter=",")
    
    save_data = np.column_stack((wl_fit, np.real(eps_fit) + eps_inf, np.imag(eps_fit),
                                  np.real(model_eps), np.imag(model_eps)))
    eps_save = os.path.join(save_path, "eps_fit_sample_data.csv")
    np.savetxt(eps_save, save_data, delimiter=",",
               header="wavelength(um),Material_Re,Material_Im,Meep_Re,Meep_Im", comments="")
    print(f"계산된 meep material fitting 정보 CSV 파일 저장 완료.")


Material_fit(save_path="")
