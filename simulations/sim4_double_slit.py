"""Simulation 4: double-slit interference in 2D, with absorbing boundary layers."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.chdir(os.path.join(ROOT, "results"))  # all intermediate data lives in results/
import json
import numpy as np
from scipy.signal import find_peaks
from ssfm import SSFM

Nx, Ny = 1024, 512
Lx0, Lx1, Ly = -20.0, 60.0, 30.0
x = np.linspace(Lx0, Lx1, Nx, endpoint=False)
y = np.linspace(-Ly, Ly, Ny, endpoint=False)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij")
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")

k0, d, w_slit, wall_t, V_wall = 6.0, 6.0, 0.6, 0.4, 150.0
lam = 2 * np.pi / k0
D = 40.0                                  # wall at x = 0, screen at x = D

V = np.zeros_like(X, dtype=complex)
in_wall = (np.abs(X) < wall_t / 2)
in_slit = (np.abs(np.abs(Y) - d / 2) < w_slit / 2)
V[in_wall & ~in_slit] = V_wall
# absorbing layers (negative imaginary potential) on every edge of the box
def ramp(s, width):
    return np.clip(s / width, 0, 1) ** 2
W = 4.0
cap = 3.0 * (ramp(Lx0 + W - X, W) + ramp(X - (Lx1 - W), W) + ramp(-Ly + W - Y, W) + ramp(Y - (Ly - W), W))
V = V - 1j * cap

dt = 0.004
prop = SSFM(V, KX ** 2 + KY ** 2, dt)
sx, sy = 2.0, 6.0
psi = np.exp(-(X + 10.0) ** 2 / (4 * sx ** 2) - Y ** 2 / (4 * sy ** 2) + 1j * k0 * X)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy)

ix = np.argmin(np.abs(x - D))
screen = np.zeros(Ny)
t_end = 14.0
n = int(round(t_end / dt))
snap_t = 9.0
snap = None
for i in range(n):
    psi = prop.step(psi)
    screen += np.abs(psi[ix, :]) ** 2 * dt          # time-integrated density at the screen
    if snap is None and (i + 1) * dt >= snap_t:
        snap = np.abs(psi) ** 2
    if abs((i + 1) * dt - 8.0) < dt / 2:
        np.save("sim4_psi_t8.npy", psi)
screen /= screen.max()

# predicted bright-fringe positions: exact path difference = n*lambda, and small-angle spacing
def y_exact(m):
    from scipy.optimize import brentq
    f = lambda yy: np.hypot(D, yy + d / 2) - np.hypot(D, yy - d / 2) - m * lam
    return brentq(f, 0, 200.0) if m > 0 else 0.0
pred = [y_exact(m) for m in range(0, 4) if m * lam < d * 0.98]
pred = [p for p in pred if p < Ly - 6]
spacing_small_angle = lam * D / d

central = np.abs(y) < 22
peaks, _ = find_peaks(screen * central, height=0.08, distance=int(2.0 / dy))
yp = np.sort(y[peaks])
pos = yp[yp >= -0.5]
print("numerical peaks (y>=0):", np.round(pos, 3))
print("exact path-difference peaks:", np.round(pred, 3), "small-angle spacing:", round(spacing_small_angle, 3))
match = []
for m, yy in enumerate(pred):
    j = np.argmin(np.abs(pos - yy))
    match.append(dict(m=m, y_pred=float(yy), y_num=float(pos[j]), diff=float(pos[j] - yy)))
print(match)

np.savez("sim4.npz", x=x, y=y, snap=snap, screen=screen, D=D, d=d, w=w_slit, wall_t=wall_t,
         pred=pred, peaks=yp)
json.dump(dict(k0=k0, lam=lam, d=d, w=w_slit, D=D, spacing_small_angle=spacing_small_angle,
               fringes=match, dy=float(dy)), open("sim4_results.json", "w"), indent=1)
