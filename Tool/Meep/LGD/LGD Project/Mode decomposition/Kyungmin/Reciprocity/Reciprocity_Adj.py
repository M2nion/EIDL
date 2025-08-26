# inverse_reciprocity_led.py
# Meep + Adjoint: reciprocity 기반 역설계 (Custom Source/Exploiting Reciprocity geometry)
# Requires: meep, meep-materials (옵션), nlopt, autograd, jax (CPU 가능)
import math
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import meep as mp
import meep.adjoint as mpa
from meep.materials import Ag
import nlopt

# ----------------------------
# Geometry & sim params (from Meep "Exploiting Reciprocity" example)
# ----------------------------
resolution = 50   # px/um (문서 예시와 동일: 높은 해상도 권장)
nfreq      = 100
fcen       = 1.0   # center frequency
df         = 0.2   # bandwidth
frequencies = np.linspace(fcen - 0.5*df, fcen + 0.5*df, nfreq)

dpml = 1.0
dair = 2.0
hrod = 0.7
wrod = 0.5
dsub = 5.0
dAg  = 0.5

sx = 1.1
sy = dpml + dair + hrod + dsub + dAg
cell_size = mp.Vector3(sx, sy)

substrate = mp.Medium(index=3.45)
air       = mp.Medium(index=1.0)

pml_layers = [mp.PML(direction=mp.Y, thickness=dpml, side=mp.High)]

# 디자인 영역(텍스처 블록 자리)을 MaterialGrid로 둔다: 기판(3.45) vs 공기(1.0) 이진
DESIGN_W = wrod
DESIGN_H = hrod
design_c = mp.Vector3(0, 0.5*sy - dpml - dair - 0.5*hrod)
design_sz= mp.Vector3(DESIGN_W, DESIGN_H)

# 디자인 격자 해상도(디자인 변수 개수) — 해상도가 높을수록 자유도↑
design_res = resolution  # 한 변당 px 수 ~ 물리해상도 수준
NX = int(DESIGN_W * design_res) + 1
NY = int(DESIGN_H * design_res) + 1

# 최소 피쳐/이진화를 위한 filter/projection 설정
MIN_FEATURE = 0.08     # (um) 최소선폭/간격 목표치 ~80 nm 예시
eta_i = 0.5            # tanh projection threshold (blueprint)
beta0 = 8              # 초기 beta (점차 증가)
beta_scale = 2         # beta 증분 스케줄
num_beta_steps = 3     # 스케줄 스텝 수
filter_radius = mpa.get_conic_radius_from_eta_e(MIN_FEATURE, 0.55)

# 목표 각도 설정(법선 포함). 다중각은 가중합으로 최적화됨.
TARGET_ANGLES_DEG = [0.0]   # 예: [0.0, 20.0, -20.0]
ANGLE_WEIGHTS     = [1.0]   # 각 각도의 가중치(합 1 권장)

assert len(TARGET_ANGLES_DEG) == len(ANGLE_WEIGHTS)

# ----------------------------
# 디자인 변수/영역 정의
# ----------------------------
design_vars = mp.MaterialGrid(
    mp.Vector3(NX, NY, 0),
    air, substrate,
    grid_type="U_MEAN",  # 평균 혼합 (subpixel averaging과 호환)
    beta=0,              # subpixel averaging은 뒤에서 projection과 함께 활성화
    do_averaging=False
)
design_region = mpa.DesignRegion(
    design_vars,
    volume=mp.Volume(center=design_c, size=design_sz),
)

# 매핑: (필터+프로젝션) → 0..1 그레이스케일 → MaterialGrid 내부에서 재료보간
def map_weights(x_flat, eta, beta):
    x = x_flat.reshape(NX, NY)
    xf = mpa.conic_filter(x, filter_radius, DESIGN_W, DESIGN_H, design_res)
    xp = mpa.tanh_projection(xf, beta, eta)
    # 좌우 대칭을 원하면 아래 줄 사용(여기선 비활성화)
    # xp = (npa.flipud(xp) + xp) / 2
    return xp.flatten()

# ----------------------------
# Geometry builder (reciprocity backward run)
# ----------------------------
def build_geometry_with_design():
    geom = [
        # Substrate bulk
        mp.Block(
            material=substrate,
            center=mp.Vector3(0, 0.5*sy - dpml - dair - hrod - 0.5*dsub),
            size=mp.Vector3(mp.inf, dsub, mp.inf),
        ),
        # Ag back reflector
        mp.Block(
            material=Ag,
            center=mp.Vector3(0, -0.5*sy + 0.5*dAg),
            size=mp.Vector3(mp.inf, dAg, mp.inf),
        ),
        # Design region block (텍스처 블록 자리)
        mp.Block(
            material=design_vars,
            center=design_region.center,
            size=design_region.size,
        ),
    ]
    return geom

# ----------------------------
# 한 각도(theta)용 시뮬+목적량 설정
# ----------------------------
def make_problem_for_angle(theta_deg):
    theta = math.radians(theta_deg)

    # backward: 공기측(top)에서 아래(-y)로 진행하는 평면파 소스
    # 문서 예시처럼 line source를 상단 air에 배치 (is_integrated 불필요: PML 바깥)
    src = [mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(0, 0.5*sy - dpml),
        size=mp.Vector3(sx, 0)
    )]

    geometry = build_geometry_with_design()

    # k_point 설정:
    #   broadband에서는 보통 fmin 기준으로 kx를 잡는다 (문서 권장)
    fmin = float(np.min(frequencies))
    nx_medium = 1.0  # 상부는 공기
    kx = nx_medium * fmin * math.sin(theta)  # Meep convention
    kpoint = mp.Vector3(kx, 0, 0) if abs(theta) > 1e-9 else mp.Vector3()

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=src,
        k_point=kpoint
    )

    # 소스층(기판 내부 라인)에서 DFT 필드 모니터(=adjoint 목적량용 FourierFields)
    srcline_center = mp.Vector3(0, -0.5*sy + dAg + 0.5*dsub)
    srcline_size   = mp.Vector3(sx, 0, 0)
    FF = mpa.FourierFields(sim, mp.Volume(center=srcline_center, size=srcline_size), mp.Ez)

    # 목적함수: 라인 전체/대역 평균의 |Ez|^2 (상호성에 따른 단일 방향 방사 증대에 해당)
    def J(fields):  # fields shape: (nfreq, Nx_line)
        return npa.mean(npa.abs(fields)**2)

    # OptimizationProblem 구성
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J],
        objective_arguments=[FF],
        design_regions=[design_region],
        frequencies=frequencies,
        maximum_run_time=4000,  # 안전 마진
    )
    return opt

# 각도별 문제 사전 생성(동일 design_region 공유)
PROBLEMS = [make_problem_for_angle(th) for th in TARGET_ANGLES_DEG]

# ----------------------------
# NLopt 루프: 각도 가중합 최대화
# ----------------------------
history = []
def objective_wrapper(v, grad_out, cur_beta):
    # MaterialGrid 내부에서 subpixel averaging을 쓸 거면 beta>0로 설정
    design_vars.beta = cur_beta
    design_vars.do_averaging = True

    # 맵핑(필터/프로젝션) 적용
    mapped = map_weights(v, eta_i, cur_beta)

    # 각도별 목적/그라디언트 계산 후 가중합
    total_J = 0.0
    total_dJdu = 0.0

    for w, opt in zip(ANGLE_WEIGHTS, PROBLEMS):
        Jval, dJ_du = opt([mapped])  # forward+adjoint 자동 수행
        total_J += w * np.real(Jval)
        total_dJdu = (total_dJdu + w * dJ_du[0]) if np.size(total_dJdu) else (w * dJ_du[0])

    # 역전파: dJ/dv = dJ/du · du/dv
    if grad_out.size > 0:
        vjp = tensor_jacobian_product(map_weights, 0)(v, eta_i, cur_beta, np.sum(total_dJdu, axis=1))
        grad_out[:] = np.real(vjp)

    history.append(float(np.real(total_J)))
    return float(np.real(total_J))

# ----------------------------
# 실행부
# ----------------------------
if __name__ == "__main__":
    n_vars = NX * NY
    x0 = np.ones((n_vars,)) * 0.5  # 중립 초기화

    # 경사상승 (maximize) — MMA 사용
    algorithm = nlopt.LD_MMA
    lb = np.zeros_like(x0)
    ub = np.ones_like(x0)

    x = x0.copy()
    beta = beta0
    maxeval_per_beta = 30
    ftol = 1e-4

    for _ in range(num_beta_steps):
        solver = nlopt.opt(algorithm, n_vars)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda a, g: objective_wrapper(a, g, beta))
        solver.set_maxeval(maxeval_per_beta)
        solver.set_ftol_rel(ftol)
        x[:] = solver.optimize(x)
        beta *= beta_scale  # 이진화 강화

    # 최종 디자인 저장 (0..1 그레이스케일)
    final_mapped = map_weights(x, eta_i, beta/beta_scale)
    np.save("optimized_weights.npy", final_mapped.reshape(NX, NY))
    if mp.am_master():
        print("Done. Best J =", history[-1])
        print("Saved: optimized_weights.npy")
