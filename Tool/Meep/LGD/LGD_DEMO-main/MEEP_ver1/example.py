import meep as mp
from meep.materials import Al as ALU
import numpy as np

lambda_min = 0.4       # 최소 파장 (µm)
lambda_max = 0.8       # 최대 파장 (µm)
fmin = 1/lambda_max    # 최소 주파수
fmax = 1/lambda_min    # 최대 주파수
fcen = 0.5*(fmin+fmax) # 중앙 주파수
df = fmax-fmin         # 주파수 대역폭

resolution = 55        # 시뮬레이션 해상도
nfreq = 25             # 추출할 주파수 개수                                                             

tABS = 0.5    # X/Y 방향 흡수 경계층 두께
tPML = 0.5    # Z 방향 PML 경계층 두께
tGLS = 0.5    # 글래스 층 두께
tITO = 0.5    # ITO 층 두께
tORG = 0.5    # 유기층(OLED 발광층) 두께
tALU = 0.2    # 알루미늄(캐소드) 두께

L = 1.0       # OLED의 가로/세로 길이

sz = tPML + tGLS + tITO + tORG + tALU  # Z 방향 전체 길이
sxy = L + 2*tABS                        # X/Y 방향 길이
cell_size = mp.Vector3(sxy, sxy, sz)    # 시뮬레이션 셀 크기

boundary_layers = [mp.Absorber(tABS,direction=mp.X),
                   mp.Absorber(tABS,direction=mp.Y),
                   mp.PML(tPML,direction=mp.Z,side=mp.High)]

ORG = mp.Medium(index=1.75)   # OLED 유기층의 굴절률 설정

geometry = [
    mp.Block(material=mp.Medium(index=1.5),      # 유리층
             size=mp.Vector3(mp.inf, mp.inf, tPML + tGLS),
             center=mp.Vector3(z=0.5*sz - 0.5*(tPML + tGLS))),
    
    mp.Block(material=mp.Medium(index=1.2),      # ITO 층
             size=mp.Vector3(mp.inf, mp.inf, tITO),
             center=mp.Vector3(z=0.5*sz - tPML - tGLS - 0.5*tITO)),
    
    mp.Block(material=ORG,                       # 유기층
             size=mp.Vector3(mp.inf, mp.inf, tORG),
             center=mp.Vector3(z=0.5*sz - tPML - tGLS - tITO - 0.5*tORG)),
    
    mp.Block(material=ALU,                       # 알루미늄 층
             size=mp.Vector3(mp.inf, mp.inf, tALU),
             center=mp.Vector3(z=0.5*sz - tPML - tGLS - tITO - tORG - 0.5*tALU))
]

sources = [
    mp.Source(mp.ContinuousSource(fcen),     # 연속파 광원
              component=mp.Ez,                # Z 방향 전기장
              center=mp.Vector3(z=0.5*sz - tPML - tGLS - tITO - 0.5*tORG))
]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry,
                    sources=sources,
                    eps_averaging=False)
                    
# surround source with a six-sided box of flux planes                                                             
srcbox_width = 0.05
srcbox_top = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=0.5*sz-tPML-tGLS), size=mp.Vector3(srcbox_width,srcbox_width,0), direction=mp.Z, weight=+1))
srcbox_bot = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=0.5*sz-tPML-tGLS-tITO-0.8*tORG), size=mp.Vector3(srcbox_width,srcbox_width,0), direction=mp.Z, weight=-1))
srcbox_xp = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0.5*srcbox_width,0,0.5*sz-tPML-tGLS-0.5*(tITO+0.8*tORG)), size=mp.Vector3(0,srcbox_width,tITO+0.8*tORG), direction=mp.X, weight=+1))
srcbox_xm = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(-0.5*srcbox_width,0,0.5*sz-tPML-tGLS-0.5*(tITO+0.8*tORG)), size=mp.Vector3(0,srcbox_width,tITO+0.8*tORG), direction=mp.X, weight=-1))
srcbox_yp = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0,0.5*srcbox_width,0.5*sz-tPML-tGLS-0.5*(tITO+0.8*tORG)), size=mp.Vector3(srcbox_width,0,tITO+0.8*tORG), direction=mp.Y, weight=+1))
srcbox_ym = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0,-0.5*srcbox_width,0.5*sz-tPML-tGLS-0.5*(tITO+0.8*tORG)), size=mp.Vector3(srcbox_width,0,tITO+0.8*tORG), direction=mp.Y, weight=-1))

# padding for flux box to fully capture waveguide mode                                                            
fluxbox_dpad = 0.05

glass_flux = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=0.5*sz-tPML-(tGLS-fluxbox_dpad)), size = mp.Vector3(L,L,0), direction=mp.Z, weight=+1))
wvgbox_xp = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(size=mp.Vector3(0,L,fluxbox_dpad+tITO+tORG+fluxbox_dpad),direction=mp.X, center=mp.Vector3(0.5*L,0,0.5*sz-tPML-tGLS-0.5*(tITO+tORG)), weight=+1))
wvgbox_xm = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(size=mp.Vector3(0,L,fluxbox_dpad+tITO+tORG+fluxbox_dpad),direction=mp.X, center=mp.Vector3(-0.5*L,0,0.5*sz-tPML-tGLS-0.5*(tITO+tORG)), weight=-1))
wvgbox_yp = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(size=mp.Vector3(L,0,fluxbox_dpad+tITO+tORG+fluxbox_dpad),direction=mp.Y, center=mp.Vector3(0,0.5*L,0.5*sz-tPML-tGLS-0.5*(tITO+tORG)), weight=+1))
wvgbox_ym = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(size=mp.Vector3(L,0,fluxbox_dpad+tITO+tORG+fluxbox_dpad),direction=mp.Y, center=mp.Vector3(0,-0.5*L,0.5*sz-tPML-tGLS-0.5*(tITO+tORG)), weight=-1))

mp.verbosity(2)
sim.run(until=2.0)

sim.fields.reset_timers()
for _ in range(5):
    sim.fields.step()
sim.output_times('oled_timings.csv')

print("field:, {}".format(np.real(sim.get_field_point(mp.Ez, mp.Vector3(0.1235,-0.3165,0.7298)))))

flux_srcbox_top = np.asarray(mp.get_fluxes(srcbox_top))
flux_srcbox_bot = np.asarray(mp.get_fluxes(srcbox_bot))
flux_srcbox_xp = np.asarray(mp.get_fluxes(srcbox_xp))
flux_srcbox_xm = np.asarray(mp.get_fluxes(srcbox_xm))
flux_srcbox_yp = np.asarray(mp.get_fluxes(srcbox_yp))
flux_srcbox_ym = np.asarray(mp.get_fluxes(srcbox_ym))

flux_wvgbox_xp = np.asarray(mp.get_fluxes(wvgbox_xp))
flux_wvgbox_xm = np.asarray(mp.get_fluxes(wvgbox_xm))
flux_wvgbox_yp = np.asarray(mp.get_fluxes(wvgbox_yp))
flux_wvgbox_ym = np.asarray(mp.get_fluxes(wvgbox_ym))

flux_glass = np.asarray(mp.get_fluxes(glass_flux))
flux_total = flux_srcbox_top+flux_srcbox_bot+flux_srcbox_xp+flux_srcbox_xm+flux_srcbox_yp+flux_srcbox_ym
flux_waveguide = flux_wvgbox_xp+flux_wvgbox_xm+flux_wvgbox_yp+flux_wvgbox_ym

print("flux_glass:, {}".format(flux_glass))
print("flux_waveguide:, {}".format(flux_waveguide))
print("flux_total:, {}".format(flux_total))

print("sum(flux_glass):, {}".format(np.sum(flux_glass)))
print("sum(flux_waveguide):, {}".format(np.sum(flux_waveguide)))
print("sum(flux_total):, {}".format(np.sum(flux_total)))

# flux 데이터를 oled-flux.dat 파일로 저장
with open("oled-flux.dat", "w") as f:
    # 헤더 작성
    f.write("# freq flux_srcbox_top flux_srcbox_bot flux_srcbox_xp flux_srcbox_xm flux_srcbox_yp flux_srcbox_ym flux_glass flux_wvgbox_xp flux_wvgbox_xm flux_wvgbox_yp flux_wvgbox_ym\n")
    
    for i in range(nfreq):
        f.write(f"{fcen - 0.5*df + i*df/(nfreq-1):.6e} "  # 주파수
                f"{flux_srcbox_top[i]:.6e} {flux_srcbox_bot[i]:.6e} "
                f"{flux_srcbox_xp[i]:.6e} {flux_srcbox_xm[i]:.6e} "
                f"{flux_srcbox_yp[i]:.6e} {flux_srcbox_ym[i]:.6e} "
                f"{flux_glass[i]:.6e} "
                f"{flux_wvgbox_xp[i]:.6e} {flux_wvgbox_xm[i]:.6e} "
                f"{flux_wvgbox_yp[i]:.6e} {flux_wvgbox_ym[i]:.6e}\n")