"""Far-field (Fraunhofer) analysis of Simulation 4.

The transmitted part of the wavefunction (x > 1) is Fourier transformed; its momentum
distribution, integrated over |k|, gives the intensity versus angle, which is exactly the
far-field pattern. Bright fringes are expected at sin(theta_m) = m * lambda / d.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.chdir(os.path.join(ROOT, "results"))  # all intermediate data lives in results/
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

psi = np.load("sim4_psi_t8.npy")
Nx, Ny = psi.shape
x = np.linspace(-20, 60, Nx, endpoint=False)
y = np.linspace(-30, 30, Ny, endpoint=False)
dx, dy = x[1] - x[0], y[1] - y[0]
k0, d = 6.0, 6.0
lam = 2 * np.pi / k0

pt = psi * (x[:, None] > 1.0)
Px, Py = 2 * Nx, 4 * Ny                       # zero-padding refines the k grid
F = np.abs(np.fft.fftshift(np.fft.fft2(pt, s=(Px, Py)))) ** 2
kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Px, d=dx))
ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Py, d=dy))
interp = RegularGridInterpolator((kx, ky), F, bounds_error=False, fill_value=0.0)

theta = np.radians(np.linspace(-60, 60, 1201))
kr = np.linspace(3.0, 9.0, 301)
TH, KR = np.meshgrid(theta, kr, indexing="ij")
vals = interp(np.stack([KR * np.cos(TH), KR * np.sin(TH)], axis=-1))
I = np.trapezoid(vals * KR, kr, axis=1)
I /= I.max()

pk, _ = find_peaks(I, height=0.02, prominence=0.02)
ang = np.degrees(theta[pk])
pred = [float(np.degrees(np.arcsin(m * lam / d))) for m in range(0, 5) if m * lam / d < 1]
match = []
for m, a in enumerate(pred):
    j = np.argmin(np.abs(ang - a))
    match.append(dict(m=m, theta_pred=a, theta_num=float(ang[j]), diff=float(ang[j] - a)))
print("peaks (deg):", np.round(ang, 2))
print("predicted :", np.round(pred, 2))
for r in match:
    print(r)
np.savez("sim4_far.npz", theta=np.degrees(theta), I=I, pred=pred)
json.dump(dict(lam=lam, d=d, peaks=[float(a) for a in ang], fringes=match,
               transmitted_prob=float(np.sum(np.abs(pt) ** 2) * dx * dy)),
          open("sim4_far.json", "w"), indent=1)

# Transverse-momentum distribution: for two slits the fringes sit at k_y = 2*pi*m/d,
# independent of the particle's energy, which removes the packet's momentum spread.
Iky = np.sum(F[kx > 0, :], axis=0)
Iky /= Iky.max()
sel = np.abs(ky) < 6
pk2, _ = find_peaks(Iky * sel, height=0.02, prominence=0.02)
kyp = ky[pk2]
ys = y[(np.abs(np.abs(y) - d / 2) < 0.3)]
d_eff = ys[ys > 0].mean() - ys[ys < 0].mean()     # slit centres as resolved on the grid
pred_ky = [2 * np.pi * m / d_eff for m in range(0, 6) if 2 * np.pi * m / d_eff < 6]
mky = []
for m, q in enumerate(pred_ky):
    j = np.argmin(np.abs(kyp - q))
    mky.append(dict(m=m, ky_pred=float(q), ky_num=float(kyp[j]), rel=float((kyp[j] - q) / q) if q else 0.0))
print("d_eff on grid:", d_eff)
for r in mky:
    print(r)
np.savez("sim4_ky.npz", ky=ky[sel], I=Iky[sel], pred=pred_ky)
r = json.load(open("sim4_far.json"))
r.update(d_eff=float(d_eff), ky_fringes=mky, dky=float(ky[1] - ky[0]))
json.dump(r, open("sim4_far.json", "w"), indent=1)
