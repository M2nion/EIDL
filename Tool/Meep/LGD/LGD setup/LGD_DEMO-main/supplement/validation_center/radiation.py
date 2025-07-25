import math
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

import meep as mp

resolution = 50  # pixels/um

sxy = 4
dpml = 1

cell = mp.Vector3(sxy + 2 * dpml, sxy + 2 * dpml)

pml_layers = [mp.PML(dpml)]

fcen = 1.0
df = 0.4
src_cmpt = mp.Ez
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(), component=src_cmpt
    )
]

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
    fcen,
    0,
    1,
    mp.Near2FarRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(sxy, 0)),
    mp.Near2FarRegion(
        center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(sxy, 0), weight=-1
    ),
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

sim.plot2D()
if mp.am_master():
    plt.savefig("2Dplot.png", dpi=150, bbox_inches="tight")

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
        f"radiation_pattern_{mp.component_name(src_cmpt)}.png",
        dpi=150,
        bbox_inches="tight",
    )