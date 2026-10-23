"""Fast checks of the solver against exact results. Run with:  python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssfm import SSFM, grid_1d, gaussian, moments, sigma_free, T_barrier, packet_average  # noqa: E402


def free_run(t_end=2.0, dt=0.01, N=2048, L=40.0):
    x, dx, k = grid_1d(N, L)
    psi = gaussian(x, 0.0, 1.0, 2.0)
    prop = SSFM(np.zeros_like(x), k ** 2, dt)
    return x, prop.step(psi, int(round(t_end / dt)))


def test_norm_is_conserved_to_machine_precision():
    x, psi = free_run()
    norm, _, _ = moments(psi, x)
    assert abs(norm - 1) < 1e-12


def test_free_packet_width_matches_analytic_spreading():
    x, psi = free_run(t_end=2.0)
    _, xm, sigma = moments(psi, x)
    assert abs(sigma - sigma_free(2.0, 1.0)) < 1e-8
    assert abs(xm - 2.0 * 2.0) < 1e-8          # <x> = k0 t


def test_second_order_convergence_in_time_step():
    """Harmonic oscillator: halving dt should cut the error by ~4 (Strang splitting)."""
    x, dx, k = grid_1d(512, 16.0)
    V = 0.5 * x ** 2
    psi0 = gaussian(x, 3.0, np.sqrt(0.5), 0.0)

    def evolve(n, T=np.pi):                    # n steps of dt = T/n, so every run ends at exactly T
        return SSFM(V, k ** 2, T / n).step(psi0.copy(), n)

    ref = evolve(4000)
    errs = [np.sqrt(np.sum(np.abs(evolve(n) - ref) ** 2) * dx) for n in (40, 80, 160)]
    slopes = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(slopes - 2) < 0.1), slopes


def test_barrier_transmission_matches_exact_result():
    """Gaussian packet through a square barrier vs packet-averaged plane-wave T."""
    N, L = 16384, 256.0
    x, dx, k = grid_1d(N, L)                    # dx = 1/32: barrier edges land on grid points
    V0, a, k0, s0 = 1.0, 0.5, 1.4, 20.0
    # half-open interval: exactly 32 cells = width 2a. Using |x| <= a adds a 33rd cell
    # and makes the barrier ~3% too wide, which shows up as a ~0.015 error in T.
    V = np.where((x >= -a) & (x < a), V0, 0.0)
    psi = gaussian(x, -100.0, s0, k0)
    prop = SSFM(V, k ** 2, 0.05)
    psi = prop.step(psi, int(round(140 / 0.05)))
    T_num = np.sum(np.abs(psi[x > a]) ** 2) * dx
    T_exact = packet_average(lambda kk: T_barrier(kk ** 2 / 2, V0, a), k0, s0)
    assert abs(T_num - T_exact) < 2e-3, (T_num, T_exact)
