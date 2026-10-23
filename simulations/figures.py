"""Publication figures for the paper (PDF) plus PNG copies for LinkedIn."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.chdir(os.path.join(ROOT, "results"))  # all intermediate data lives in results/
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ssfm import sigma_free, R_step

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "savefig.bbox": "tight",
})
INK, NUM, ACC, GREY = "#1b1f24", "#c0392b", "#1f5fa8", "#8a8f98"
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(ROOT, "figures", f"{name}.pdf"))
    fig.savefig(os.path.join(ROOT, "figures", f"{name}.png"), dpi=200)
    plt.close(fig)

# ---- Figure 1: sigma(t) and norm ----
a, b = np.load("sim1_L20.npz"), np.load("sim1_L40.npz")
tf = np.linspace(0, 4, 400)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax[0].plot(tf, sigma_free(tf, 1.0), color=INK, label="analytic")
ax[0].plot(a["t"], a["sigma"], "o", ms=3.2, mfc="none", color=NUM, label=r"SSFM, $L=20\,a_0$")
ax[0].plot(b["t"], b["sigma"], "s", ms=2.4, color=ACC, label=r"SSFM, $L=40\,a_0$")
ax[0].set_xlabel(r"$t$ (a.u.)"); ax[0].set_ylabel(r"$\sigma(t)\ (a_0)$"); ax[0].legend(frameon=False)
ax[1].semilogy(a["t"], np.abs(a["norm"] - 1) + 1e-17, color=NUM, label=r"$L=20$")
ax[1].semilogy(b["t"], np.abs(b["norm"] - 1) + 1e-17, color=ACC, label=r"$L=40$")
ax[1].set_xlabel(r"$t$ (a.u.)"); ax[1].set_ylabel(r"$|\,\int|\psi|^2dx-1\,|$")
ax[1].set_ylim(1e-17, 1e-12); ax[1].legend(frameon=False)
save(fig, "fig1_sigma_norm")

# ---- Figure 2: snapshots ----
fig, ax = plt.subplots(figsize=(7.2, 2.5))
cols = plt.cm.viridis(np.linspace(0.1, 0.85, 5))
for t, c in zip(range(5), cols):
    ax.plot(b["x"], b[f"snap{t}"], color=c, label=fr"$t={t}$")
ax.set_xlim(-14, 22); ax.set_xlabel(r"$x\ (a_0)$"); ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.legend(frameon=False, ncol=5, loc="upper right")
save(fig, "fig2_snapshots")

# ---- Figure 3: barrier transmission ----
s2 = np.load("sim2.npz"); r2 = json.load(open("sim235_results.json"))["sim2"]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(s2["E"], s2["T"], color=INK, label="exact plane wave")
ax.plot(s2["E_num"], s2["T_num"], "o", ms=4, mfc="none", color=NUM, label="SSFM wavepacket")
ax.axvline(r2["V0"], color=GREY, ls=":", lw=1)
ax.text(r2["V0"] + 0.03, 0.08, r"$E=V_0$", color=GREY, fontsize=8.5)
ax.set_xlabel(r"$E\ (E_h)$"); ax.set_ylabel(r"transmission $T$"); ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper left")
save(fig, "fig3_barrier")

# ---- Figure 4: coherent state + convergence ----
s3 = np.load("sim3.npz"); cv = np.load("sim3_conv.npz")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.5, 1]))
tt = np.linspace(0, s3["t"][-1], 600)
ax[0].plot(tt, 3 * np.cos(tt), color=INK, label=r"classical $A\cos\omega t$")
ax[0].plot(s3["t"][::6], s3["x"][::6], "o", ms=3, mfc="none", color=NUM, label=r"SSFM $\langle x\rangle$")
ax[0].set_xlabel(r"$t$ (a.u.)"); ax[0].set_ylabel(r"$\langle x\rangle\ (a_0)$"); ax[0].set_ylim(-3.4, 4.6)
ax[0].legend(frameon=False, loc="upper center", ncol=2)
dts, err = cv["dt"], cv["err"]
ax[1].loglog(dts, err, "o-", color=NUM, ms=4, label="measured")
ax[1].loglog(dts, err[0] * (dts / dts[0]) ** 2, "--", color=GREY, lw=1, label=r"$\propto\Delta t^2$")
ax[1].set_xlabel(r"$\Delta t$"); ax[1].set_ylabel(r"$L^2$ error at $t=\pi$"); ax[1].legend(frameon=False)
save(fig, "fig4_coherent")

# ---- Figure 5: double slit ----
s4 = np.load("sim4.npz"); ky = np.load("sim4_ky.npz")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
x, y = s4["x"], s4["y"]
ax[0].imshow(np.sqrt(s4["snap"]).T, extent=[x[0], x[-1], y[0], y[-1]], origin="lower",
             aspect="auto", cmap="magma")
for yy in (-30, s4["d"] / 2 + s4["w"] / 2):
    pass
ax[0].axvline(0, color="white", lw=0.6, alpha=0.6)
ax[0].set_xlim(-10, 56); ax[0].set_ylim(-26, 26)
ax[0].set_xlabel(r"$x\ (a_0)$"); ax[0].set_ylabel(r"$y\ (a_0)$")
ax[1].plot(ky["ky"], ky["I"], color=INK)
for m, q in enumerate(ky["pred"]):
    for sgn in ([1] if m == 0 else [1, -1]):
        ax[1].axvline(sgn * q, color=NUM, ls=":", lw=1)
ax[1].set_xlabel(r"$k_y\ (a_0^{-1})$"); ax[1].set_ylabel("transmitted intensity (norm.)")
ax[1].set_xlim(-5.6, 5.6)
save(fig, "fig5_double_slit")

# ---- Figure 6: step potential ----
s5 = np.load("sim5.npz"); r5 = json.load(open("sim235_results.json"))["sim5"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.3, 1]))
xx = s5["x"]
ax[0].plot(xx, s5["p0"], color=GREY, label=r"$t=0$")
ax[0].plot(xx, s5["p1"], color=ACC, label="after scattering")
ax[0].axvspan(0, 120, color=GREY, alpha=0.12, lw=0)
ax[0].text(3, 0.036, r"$V=V_0$", color=GREY, fontsize=8.5)
ax[0].set_xlim(-140, 120); ax[0].set_xlabel(r"$x\ (a_0)$"); ax[0].set_ylabel(r"$|\psi|^2$")
ax[0].legend(frameon=False, loc="upper left")
EE = np.linspace(1.05, 5, 400)
ax[1].semilogy(EE, R_step(EE, r5["V0"]), color=INK, label="exact")
ax[1].semilogy([r["E"] for r in r5["rows"]], [r["R_num"] for r in r5["rows"]], "o", ms=4,
               mfc="none", color=NUM, label="SSFM")
ax[1].set_xlabel(r"$E\ (E_h)$"); ax[1].set_ylabel(r"reflection $R$"); ax[1].legend(frameon=False)
save(fig, "fig6_step")
print("figures written")
