# 필수 라이브러리 임포트
from typing import Tuple
import matplotlib
import meep as mp
import nlopt
import numpy as np
import os

# matplotlib의 백엔드를 "agg"로 설정하여 이미지를 파일로 저장 가능하게 합니다.
import matplotlib.pyplot as plt
 
# Lorentzian 함수 정의: Lorentzian 파라미터를 사용하여 복소 유전율 프로파일을 계산
def lorentzfunc(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    N = len(p) // 3  # 파라미터가 3개씩 묶인 Lorentzian 항의 개수
    y = np.zeros(len(x))  # 결과값을 저장할 배열
    for n in range(N):
        A_n = p[3 * n + 0]  # 첫 번째 파라미터 (진폭)
        x_n = p[3 * n + 1]  # 두 번째 파라미터 (중심 주파수)
        g_n = p[3 * n + 2]  # 세 번째 파라미터 (감쇠율)
        y = y + A_n / (np.square(x_n) - np.square(x) - 1j * x * g_n)  # Lorentzian 함수 계산
    return y

# 잔차 함수 정의: 실제 값과 Lorentzian 모델 간의 차이를 계산하고 그라디언트도 반환
def lorentzerr(p: np.ndarray, x: np.ndarray, y: np.ndarray, grad: np.ndarray) -> float:
    N = len(p) // 3  # Lorentzian 항의 개수
    yp = lorentzfunc(p, x)  # 예측된 유전율 프로파일 계산
    val = np.sum(np.square(abs(y - yp)))  # 실제 값과 예측 값의 차이(L2 norm)

    # 그라디언트 계산
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        d = 1 / (np.square(x_n) - np.square(x) - 1j * x * g_n)  # Lorentzian 함수의 도함수
        if grad.size > 0:
            grad[3 * n + 0] = 2 * np.real(np.dot(np.conj(yp - y), d))  # 진폭에 대한 그라디언트
            grad[3 * n + 1] = -4 * x_n * A_n * np.real(np.dot(np.conj(yp - y), np.square(d)))  # 중심 주파수에 대한 그라디언트
            grad[3 * n + 2] = -2 * A_n * np.imag(np.dot(np.conj(yp - y), x * np.square(d)))  # 감쇠율에 대한 그라디언트
    return val  # 최종 오차 반환

# Lorentzian 파라미터 최적화 함수 정의
def lorentzfit(
    p0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alg=nlopt.LD_LBFGS,
    tol: float = 1e-25,
    maxeval: float = 10000,
) -> Tuple[np.ndarray, float]:
    # NLopt 최적화 설정
    opt = nlopt.opt(alg, len(p0))  # 최적화 알고리즘 설정 (LD_LBFGS는 LBFGS 알고리즘)
    opt.set_ftol_rel(tol)  # 상대 오차 허용 범위 설정
    opt.set_maxeval(maxeval)  # 최대 반복 횟수 설정
    opt.set_lower_bounds(np.zeros(len(p0)))  # 하한값 설정
    opt.set_upper_bounds(float("inf") * np.ones(len(p0)))  # 상한값 설정
    opt.set_min_objective(lambda p, grad: lorentzerr(p, x, y, grad))  # 목표 함수 설정
    local_opt = nlopt.opt(nlopt.LD_LBFGS, len(p0))  # 로컬 최적화 설정
    local_opt.set_ftol_rel(1e-10)  # 로컬 최적화 상대 오차 설정
    local_opt.set_xtol_rel(1e-8)  # 로컬 최적화 X 변화 상대 오차 설정
    opt.set_local_optimizer(local_opt)  # 로컬 최적화 설정 추가
    popt = opt.optimize(p0)  # 최적화 실행
    minf = opt.last_optimum_value()  # 마지막 최적화 값 반환
    return popt, minf  # 최적화된 파라미터와 최소 오차 반환