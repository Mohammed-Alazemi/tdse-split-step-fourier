"""Reproduce every number and figure in the paper.

    python run_all.py          # all five simulations + figures (a few minutes; the 2D run is the slowest)
    python run_all.py --quick  # skip the 2D double slit
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["sim1_free_particle.py", "sim2_3_5_barrier_oscillator_step.py",
           "sim4_double_slit.py", "sim4_far_field.py", "figures.py"]

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    for name in SCRIPTS:
        if quick and name.startswith("sim4"):
            continue
        if quick and name == "figures.py" and not os.path.exists(os.path.join(HERE, "results", "sim4.npz")):
            print("skipping figures.py (needs the double-slit results; run without --quick)")
            continue
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "simulations", name)], check=True)
        print(f"--- {name} done in {time.time() - t0:.1f} s", flush=True)
