"""Simulations 2, 3 and 5: barrier tunnelling, harmonic-oscillator coherent state, step potential."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.chdir(os.path.join(ROOT, "results"))  # all intermediate data lives in results/
import json
import numpy as np
from ssfm import (grid_1d, gaussian, SSFM, moments, T_barrier, R_step, packet_average)

res = {}

# ---------- Simulation 2: rectangular barrier, transmission vs energy ----------
N, L = 16384, 256.0                  # dx = 1/32 a0, so the barrier spans exactly 32 cells
x, dx, k = grid_1d(N, L)
V0, a = 1.0, 0.5                     # barrier height 1 Eh, width 2a = 1 a0
sigma0, x0, dt = 10.0, -100.0, 0.02  # wide packet -> narrow momentum spread
V = np.where((x >= -a) & (x < a), V0, 0.0)
assert abs(np.sum(V > 0) * dx - 2 * a) < 1e-12
prop = SSFM(V, k ** 2, dt)
rows = []
for k0 in np.round(np.arange(0.6, 2.01, 0.1), 2):
    E0 = 0.5 * k0 ** 2
    psi = gaussian(x, x0, sigma0, k0)
    t_end = 2 * abs(x0) / k0             # transmitted part ends ~100 a0 past the barrier
    psi = prop.step(psi, int(round(t_end / dt)))
    T_num = np.sum(np.abs(psi[x > a]) ** 2) * dx
    T_pw = float(T_barrier(np.array([E0]), V0, a)[0])
    T_av = packet_average(lambda kk: T_barrier(0.5 * kk ** 2, V0, a), k0, sigma0)
    rows.append(dict(k0=float(k0), E=E0, T_num=float(T_num), T_planewave=T_pw, T_packet=float(T_av),
                     abs_err=float(abs(T_num - T_av))))
    print(f"k0={k0:.1f} E={E0:.3f} T_num={T_num:.5f} T_pw={T_pw:.5f} T_avg={T_av:.5f}")
res["sim2"] = dict(V0=V0, a=a, sigma0=sigma0, rows=rows,
                   max_abs_err=max(r["abs_err"] for r in rows))
Ek = np.linspace(0.05, 2.2, 600)
np.savez("sim2.npz", E=Ek, T=T_barrier(Ek, V0, a),
         E_num=[r["E"] for r in rows], T_num=[r["T_num"] for r in rows],
         T_pack=[r["T_packet"] for r in rows])

# ---------- Simulation 3: harmonic-oscillator coherent state ----------
N, L = 1024, 20.0
x, dx, k = grid_1d(N, L)
w, A = 1.0, 3.0
V = 0.5 * w ** 2 * x ** 2
sig_cs = np.sqrt(1 / (2 * w))
def run_ho(dt, t_end):
    prop = SSFM(V, k ** 2, dt)
    psi = gaussian(x, A, sig_cs, 0.0)
    n = int(round(t_end / dt))
    rec = max(1, int(round(0.05 / dt)))
    ts, xs, ss, ns = [], [], [], []
    for i in range(n + 1):
        if i % rec == 0:
            nm, xm, s = moments(psi, x)
            ts.append(i * dt); xs.append(xm); ss.append(s); ns.append(nm)
        if i < n:
            psi = prop.step(psi)
    return psi, np.array(ts), np.array(xs), np.array(ss), np.array(ns)

t_end = 4 * np.pi
psi, ts, xs, ss, ns = run_ho(0.01, t_end)
dev_x = np.max(np.abs(xs - A * np.cos(w * ts)))
dev_s = np.max(np.abs(ss - sig_cs))
print(f"HO: max|<x>-A cos t|={dev_x:.2e}  max|sigma-sigma_cs|={dev_s:.2e}  norm dev={np.max(np.abs(ns-1)):.1e}")
np.savez("sim3.npz", t=ts, x=xs, s=ss)

# time-step convergence: L2 error at t = pi against a reference run with dt = 1e-4
t_c = np.pi
ref, *_ = run_ho(t_c / 31416, t_c)
errs = []
dts = [t_c / n for n in (20, 40, 80, 160, 320)]
for d in dts:
    p, *_ = run_ho(d, t_c)
    e = np.sqrt(np.sum(np.abs(p - ref) ** 2) * dx)
    errs.append(e)
slopes = np.diff(np.log(errs)) / np.diff(np.log(dts))
print("dt convergence:", [f"{e:.2e}" for e in errs], "slopes", np.round(slopes, 3))
res["sim3"] = dict(A=A, omega=w, sigma_cs=sig_cs, max_dev_x=float(dev_x), max_dev_sigma=float(dev_s),
                   norm_dev=float(np.max(np.abs(ns - 1))), dts=dts, l2_errs=[float(e) for e in errs],
                   slopes=[float(s) for s in slopes])
np.savez("sim3_conv.npz", dt=dts, err=errs)

# ---------- Simulation 5: step potential, partial reflection above the step ----------
N, L = 16384, 300.0
x, dx, k = grid_1d(N, L)
V0, sigma0, x0, dt = 1.0, 10.0, -100.0, 0.01
V = np.where(x >= 0, V0, 0.0)
prop = SSFM(V, k ** 2, dt)
rows = []
for k0 in [1.6, 1.8, 2.0, 2.5, 3.0]:
    E0 = 0.5 * k0 ** 2
    psi = gaussian(x, x0, sigma0, k0)
    psi = prop.step(psi, int(round(1.6 * abs(x0) / k0 / dt)))
    R_num = np.sum(np.abs(psi[x < 0]) ** 2) * dx
    R_pw = float(R_step(E0, V0))
    R_av = packet_average(lambda kk: np.where(0.5 * kk ** 2 > V0, R_step(np.maximum(0.5 * kk ** 2, V0 + 1e-12), V0), 1.0), k0, sigma0)
    rows.append(dict(k0=k0, E=E0, R_num=float(R_num), R_planewave=R_pw, R_packet=float(R_av),
                     abs_err=float(abs(R_num - R_av))))
    print(f"step k0={k0} E={E0:.3f} R_num={R_num:.5f} R_pw={R_pw:.5f} R_avg={R_av:.5f}")
res["sim5"] = dict(V0=V0, sigma0=sigma0, rows=rows, max_abs_err=max(r["abs_err"] for r in rows))
# snapshot pair for the figure at k0 = 2.0
psi = gaussian(x, x0, sigma0, 2.0)
p0 = np.abs(psi) ** 2
psi = prop.step(psi, int(round(1.6 * abs(x0) / 2.0 / dt)))
np.savez("sim5.npz", x=x, p0=p0, p1=np.abs(psi) ** 2, V0=V0)

json.dump(res, open("sim235_results.json", "w"), indent=1)
