#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- 안전한 백엔드(헤드리스/CLI에서도 흰 이미지 방지) ---
import matplotlib
matplotlib.use("Agg")

from typing import Optional, Dict
import os, json
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from meep.materials import Ag
dd
# ==============================
# 전역 파라미터
# ==============================
resolution = 10  # pixels/μm

nfreq    = 100   # number of frequencies
ndipole  = 11     # number of point dipoles in forward simulation

fcen = 1.0       # center frequency of Gaussian source/monitors
df   = 0.2       # frequency bandwidth of source/monitors

dpml = 1.0       # PML thickness
dair = 2.0       # air padding thickness
hrod = 0.7       # grating height
wrod = 0.5       # grating width
dsub = 5.0       # substrate thickness
dAg  = 0.5       # Ag back reflector thickness

sx = 1.1
sy = dpml + dair + hrod + dsub + dAg

cell_size  = mp.Vector3(sx, sy)
pml_layers = [mp.PML(direction=mp.Y, thickness=dpml, side=mp.High)]

# 결과 저장 루트 폴더 (해상도/디폴 수 포함)

# === X축/저장 형태 선택 ===
# 'frequency'  또는  'wavelength'
X_AXIS_MODE = 'wavelength'        # 필요 시 'frequency' 로 변경
INVERT_WAVELENGTH_AXIS = False    # 파장 그래프에서 짧은 λ을 왼쪽에 두고 싶으면 True

OUT_ROOT = f"Result/runs_res{resolution}_ndipole{ndipole}_{X_AXIS_MODE}"


# ==============================
# 지오메트리 생성
# ==============================
def substrate_geometry(is_textured: bool):
    """LED 유사 구조의 기판/은 거울/텍스처(선폭 wrod, 높이 hrod) 생성"""
    geometry = [
        mp.Block(
            material=mp.Medium(index=3.45),
            center=mp.Vector3(0, 0.5 * sy - dpml - dair - hrod - 0.5 * dsub),
            size=mp.Vector3(mp.inf, dsub, mp.inf),
        ),
        mp.Block(
            material=Ag,
            center=mp.Vector3(0, -0.5 * sy + 0.5 * dAg),
            size=mp.Vector3(mp.inf, dAg, mp.inf),
        ),
    ]

    if is_textured:
        geometry.append(
            mp.Block(
                material=mp.Medium(index=3.45),
                center=mp.Vector3(0, 0.5 * sy - dpml - dair - 0.5 * hrod),
                size=mp.Vector3(wrod, hrod, mp.inf),
            )
        )

    return geometry


# ==============================
# X축 변환 유틸
# ==============================
def _xaxis_from_freqs(freqs, mode='wavelength'):
    """주파수 배열 → (x값, 정렬인덱스, 컬럼이름, 축라벨, 파일접미사)"""
    f = np.asarray(freqs, dtype=float)
    if mode == 'wavelength':
        x = 1.0 / f                         # μm
        col = 'wavelength_um'
        label = 'wavelength (μm)'
        suffix = 'lambda'
    else:  # 'frequency'
        x = f
        col = 'frequency'
        label = 'frequency (1/μm)'
        suffix = 'freq'
    order = np.argsort(x)
    return x[order], order, col, label, suffix


# ==============================
# 저장 유틸
# ==============================
def _save_outputs(folder: str,
                  tag: str,
                  freqs: np.ndarray,
                  flux: np.ndarray,
                  sim: mp.Simulation,
                  extra_meta: Optional[Dict] = None,
                  dpi: int = 150,
                  x_mode: Optional[str] = None):
    """스펙트럼/필드 이미지/메타데이터 저장"""
    os.makedirs(folder, exist_ok=True)
    if x_mode is None:
        x_mode = X_AXIS_MODE

    # --- X축 변환/정렬 ---
    x_sorted, order, colname, xlabel, suffix = _xaxis_from_freqs(freqs, x_mode)
    flux_sorted = np.asarray(flux)[order]

    # --- 원시 데이터 ---
    np.save(os.path.join(folder, f"{colname}.npy"), x_sorted)
    np.save(os.path.join(folder, "flux.npy"),        flux_sorted)
    np.savetxt(os.path.join(folder, f"spectrum_{suffix}.csv"),
               np.c_[x_sorted, flux_sorted], delimiter=",",
               header=f"{colname},flux", comments="")

    if mp.am_master():
        # --- 스펙트럼 플롯 ---
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if np.any(flux_sorted > 0):
            ax1.semilogy(x_sorted, flux_sorted, "-")
        else:
            ax1.plot(x_sorted, flux_sorted, "-")  # 모두 0이면 semilogy 안 보일 수 있음
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("flux (|alpha|^2 or Σ|Ez|^2)")
        ax1.set_title(f"Spectrum — {tag}")
        if x_mode == 'wavelength' and INVERT_WAVELENGTH_AXIS:
            ax1.invert_xaxis()
        fig1.tight_layout(); fig1.canvas.draw()
        fig1.savefig(os.path.join(folder, f"spectrum_{suffix}.png"),
                     dpi=dpi, bbox_inches="tight")
        plt.close(fig1)

        # --- 2D 필드 스냅샷 ---
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sim.plot2D(ax=ax2)
        fig2.tight_layout(); fig2.canvas.draw()
        fig2.savefig(os.path.join(folder, "field2d.png"),
                     dpi=dpi, bbox_inches="tight")
        plt.close(fig2)

    # --- 메타데이터 ---
    meta = dict(
        x_axis_mode=x_mode,
        resolution=resolution, nfreq=nfreq, ndipole=ndipole,
        fcen=fcen, df=df, dpml=dpml, dair=dair, hrod=hrod, wrod=wrod, dsub=dsub, dAg=dAg,
        sx=sx, sy=sy,
        pml={"direction": "Y", "side": "High"},
        note="forward: |alpha|^2 of eigenmode#1 (ODD_Z); backward: Σ|Ez|^2 on line",
    )
    if extra_meta: meta.update(extra_meta)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ==============================
# Forward: dipole → air(+y)
# ==============================
def forward(n: int, rt: int, is_textured: bool):
    """
    공기(+y) 방향 모드(#1, ODD_Z)의 파워 스펙트럼(|alpha|^2) 계산 및 저장
    n: dipole 인덱스
    rt: 소스 종료 후 런타임 (nfreq/df 단위의 배수)
    is_textured: 텍스처 유무
    """
    print("----------------------------------- Forward -----------------------------------")

    # dipole x 위치 (기존 로직 유지)
    x_dip = sx * (-0.5 + n / ndipole)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(x_dip, -0.5 * sy + dAg + 0.5 * dsub),
        )
    ]
    geometry = substrate_geometry(is_textured)

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        k_point=mp.Vector3(),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
    )

    # 공기 상단에 플럭스 모니터
    flux_mon = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0.5 * sy - dpml), size=mp.Vector3(sx))
    )

    run_time = rt * nfreq / df
    sim.run(until_after_sources=run_time)

    # TM(Ez) 채널(ODD_Z)의 1번 모드 정방향 계수
    res   = sim.get_eigenmode_coefficients(flux_mon, [1], eig_parity=mp.ODD_Z)
    flux  = np.abs(res.alpha[0, :, 0]) ** 2
    freqs = mp.get_flux_freqs(flux_mon)

    # 저장
    tex_tag = "tex" if is_textured else "flat"
    tag     = f"forward_n{n:03d}_x{(x_dip):+0.3f}um_{tex_tag}"
    out_dir = os.path.join(OUT_ROOT, "forward", f"n{n:03d}_x{(x_dip):+0.3f}um_{tex_tag}")
    _save_outputs(
        folder=out_dir,
        tag=tag,
        freqs=freqs,
        flux=flux,
        sim=sim,
        extra_meta=dict(
            kind="forward",
            dipole_index=int(n),
            dipole_x_um=float(x_dip),
            textured=bool(is_textured),
            eig_parity="ODD_Z",
            eigenmode_index=1,
            run_time_units="(nfreq/df)",
            run_time=float(rt),
        ),
        x_mode=X_AXIS_MODE,
    )

    return freqs, flux


# ==============================
# Backward: plane wave(Ez) in air(−y) → substrate
# ==============================
def backward(rt: int, is_textured: bool):
    """
    공기에서 TM(Ez) 평면파를 -y로 입사시키고,
    기판 내부 라인 모니터의 DFT |Ez|^2를 적분해 스펙트럼을 저장
    """
    print("----------------------------------- Backward -----------------------------------")

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(0, 0.5 * sy - dpml),
            size=mp.Vector3(sx, 0),
        )
    ]
    geometry = substrate_geometry(is_textured)

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        k_point=mp.Vector3(),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
    )

    # 기판 내부 라인 모니터 (DFT 필드)
    dft_mon = sim.add_dft_fields(
        [mp.Ez], fcen, df, nfreq,
        center=mp.Vector3(0, -0.5 * sy + dAg + 0.5 * dsub),
        size=mp.Vector3(sx),
    )

    run_time = rt * nfreq / df
    sim.run(until_after_sources=run_time)

    freqs = mp.get_flux_freqs(dft_mon)
    abs_flux = np.zeros(nfreq)
    for nf in range(nfreq):
        dft_ez = sim.get_dft_array(dft_mon, mp.Ez, nf)
        abs_flux[nf] = np.sum(np.abs(dft_ez) ** 2)

    # 저장
    tex_tag = "tex" if is_textured else "flat"
    tag     = f"backward_xall_{tex_tag}"
    out_dir = os.path.join(OUT_ROOT, "backward", f"xall_{tex_tag}")
    _save_outputs(
        folder=out_dir,
        tag=tag,
        freqs=freqs,
        flux=abs_flux,
        sim=sim,
        extra_meta=dict(
            kind="backward",
            line_monitor_center_um=float(-0.5 * sy + dAg + 0.5 * dsub),
            textured=bool(is_textured),
            component="Ez",
            run_time_units="(nfreq/df)",
            run_time=float(rt),
        ),
        x_mode=X_AXIS_MODE,
    )

    return freqs, abs_flux


# ==============================
# 메인 루틴
# ==============================
if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    print("OUT_ROOT:", os.path.abspath(OUT_ROOT))

    # Forward: flat / textured 각각 수행 후 dipole 평균 정규화
    fwd_flat_flux = np.zeros((nfreq, ndipole))
    fwd_text_flux = np.zeros((nfreq, ndipole))
    for d in range(ndipole):
        fwd_freqs, fwd_flat_flux[:, d] = forward(d, 2, False)
        _,           fwd_text_flux[:, d] = forward(d, 4, True)

    fwd_norm_flux = np.mean(fwd_text_flux, axis=1) / np.mean(fwd_flat_flux, axis=1)

    # Backward: flat / textured
    bwd_freqs, bwd_flat_flux = backward(2, False)
    _,           bwd_text_flux = backward(4, True)
    bwd_norm_flux = bwd_text_flux / bwd_flat_flux

    # 주파수 → 선택한 X축으로 변환/정렬
    x_fwd, of, colname, xlabel, suffix = _xaxis_from_freqs(fwd_freqs, X_AXIS_MODE)
    x_bwd, ob, _,       _,      _      = _xaxis_from_freqs(bwd_freqs, X_AXIS_MODE)

    # 최종 비교 플롯 (저장)
    fig = plt.figure(figsize=(6, 4))
    if np.any(fwd_norm_flux > 0) or np.any(bwd_norm_flux > 0):
        plt.semilogy(x_fwd, fwd_norm_flux[of], "b-", label="forward")
        plt.semilogy(x_bwd, bwd_norm_flux[ob], "r-", label="backward")
    else:
        plt.plot(x_fwd, fwd_norm_flux[of], "b-", label="forward")
        plt.plot(x_bwd, bwd_norm_flux[ob], "r-", label="backward")
    plt.xlabel(xlabel)
    plt.ylabel("normalized flux")
    plt.legend()
    plt.tight_layout()
    if X_AXIS_MODE == 'wavelength' and INVERT_WAVELENGTH_AXIS:
        plt.gca().invert_xaxis()
    if mp.am_master():
        fname = os.path.join(OUT_ROOT, f"forward_vs_backward_spectrum_{suffix}_{ndipole}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
