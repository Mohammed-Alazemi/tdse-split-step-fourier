"""Split-step Fourier solver for the time-dependent Schrodinger equation.

Atomic units throughout (hbar = m = 1). Shared by all five simulations.
"""
import numpy as np


def grid_1d(N, L):
    """Periodic grid on [-L, L) and its angular wavenumbers."""
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, dx, k


def gaussian(x, x0, sigma0, k0):
    """Normalised Gaussian wavepacket with mean momentum k0."""
    psi = np.exp(-(x - x0) ** 2 / (4 * sigma0 ** 2) + 1j * k0 * x)
    dx = x[1] - x[0]
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


class SSFM:
    """Strang-split propagator: half potential kick, full kinetic kick, half potential kick.

    V may be complex: a negative imaginary part acts as an absorbing layer.
    Works for 1D or 2D arrays (k2 must be |k|^2 on the same grid).
    """

    def __init__(self, V, k2, dt, hbar=1.0, m=1.0):
        self.half_V = np.exp(-1j * V * dt / (2 * hbar))
        self.full_T = np.exp(-1j * hbar * k2 * dt / (2 * m))
        self.dt = dt

    def step(self, psi, n=1):
        fft, ifft = (np.fft.fft, np.fft.ifft) if psi.ndim == 1 else (np.fft.fft2, np.fft.ifft2)
        for _ in range(n):
            psi = self.half_V * psi
            psi = ifft(self.full_T * fft(psi))
            psi = self.half_V * psi
        return psi


def moments(psi, x):
    """Norm, <x> and width sigma from |psi|^2 (rectangle rule, exact on a periodic grid)."""
    dx = x[1] - x[0]
    p = np.abs(psi) ** 2
    norm = np.sum(p) * dx
    xm = np.sum(x * p) * dx / norm
    x2 = np.sum(x ** 2 * p) * dx / norm
    return norm, xm, np.sqrt(x2 - xm ** 2)


def sigma_free(t, sigma0, hbar=1.0, m=1.0):
    return sigma0 * np.sqrt(1 + (hbar * t / (2 * m * sigma0 ** 2)) ** 2)


def T_barrier(E, V0, a, m=1.0, hbar=1.0):
    """Exact plane-wave transmission through a barrier V0 on |x| <= a (width 2a)."""
    E = np.asarray(E, dtype=float)
    out = np.empty_like(E)
    lo, hi = E < V0, E > V0
    kap = np.sqrt(2 * m * (V0 - E[lo])) / hbar
    out[lo] = 1 / (1 + V0 ** 2 * np.sinh(2 * kap * a) ** 2 / (4 * E[lo] * (V0 - E[lo])))
    q = np.sqrt(2 * m * (E[hi] - V0)) / hbar
    out[hi] = 1 / (1 + V0 ** 2 * np.sin(2 * q * a) ** 2 / (4 * E[hi] * (E[hi] - V0)))
    eq = ~(lo | hi)
    out[eq] = 1 / (1 + m * V0 * (2 * a) ** 2 / (2 * hbar ** 2))
    return out


def R_step(E, V0, m=1.0, hbar=1.0):
    """Exact plane-wave reflection off a step V0*theta(x), valid for E > V0."""
    k1 = np.sqrt(2 * m * E) / hbar
    k2 = np.sqrt(2 * m * (E - V0)) / hbar
    return ((k1 - k2) / (k1 + k2)) ** 2


def packet_average(f, k0, sigma0, n=4001):
    """Average a plane-wave result f(k) over the packet's momentum distribution.

    |phi(k)|^2 for the Gaussian above has standard deviation 1/(2 sigma0).
    """
    sk = 1 / (2 * sigma0)
    kk = np.linspace(k0 - 8 * sk, k0 + 8 * sk, n)
    w = np.exp(-((kk - k0) ** 2) / (2 * sk ** 2))
    return np.sum(w * f(kk)) / np.sum(w)
