"""Simulation 1: free Gaussian wavepacket dispersion."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.chdir(os.path.join(ROOT, "results"))  # all intermediate data lives in results/
import json
import numpy as np
from ssfm import grid_1d, gaussian, SSFM, moments, sigma_free

x0, sigma0, k0, dt, T_total = -8.0, 1.0, 5.0, 0.01, 4.0
steps_per_unit = int(round(1 / dt))
out = {}

for label, N, L in [("L20", 1024, 20.0), ("L40", 2048, 40.0)]:
    x, dx, k = grid_1d(N, L)
    prop = SSFM(np.zeros_like(x), k ** 2, dt)
    psi = gaussian(x, x0, sigma0, k0)
    rows, ts, sig, norms, snaps = [], [], [], [], {}
    record = 20
    n_steps = int(round(T_total / dt))
    for n in range(n_steps + 1):
        t = n * dt
        if n % record == 0:
            nm, xm, s = moments(psi, x)
            ts.append(t); sig.append(s); norms.append(nm)
        if n % steps_per_unit == 0:
            nm, xm, s = moments(psi, x)
            sa = sigma_free(t, sigma0)
            rows.append(dict(t=round(t, 2), sigma_num=s, sigma_an=sa,
                             rel_err=abs(s - sa) / sa, norm=nm, mean_x=xm))
            snaps[int(round(t))] = np.abs(psi) ** 2
        if n < n_steps:
            psi = prop.step(psi)
    out[label] = dict(rows=rows, max_norm_dev=float(np.max(np.abs(np.array(norms) - 1))))
    np.savez(f"sim1_{label}.npz", x=x, t=ts, sigma=sig, norm=norms,
             **{f"snap{t}": p for t, p in snaps.items()})

json.dump(out, open("sim1_results.json", "w"), indent=1)
for label, d in out.items():
    print(label, "max |norm-1| =", d["max_norm_dev"])
    for r in d["rows"]:
        print(f"  t={r['t']:.1f} num={r['sigma_num']:.6f} an={r['sigma_an']:.6f} "
              f"err={100*r['rel_err']:.4f}% <x>={r['mean_x']:.4f} norm={r['norm']:.12f}")
