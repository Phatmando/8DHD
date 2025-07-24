#!/usr/bin/env python3
"""
prime_torus_fib_scaling.py

Torus flow in dimensions determined by Fibonacci sequence.

Features:
  • Computes toroidal flow θ_p(t) = (2π t / ln p) mod 2π for first d primes
  • d ∈ Fibonacci sequence: [1, 2, 3, 5, 8, 13, 21]
  • Visualizes each case with:
     - Parallel-coordinates
     - Random 3D projection (skipped if d < 3)
     - Composite cosine signal
     - Entropy and angular norm diagnostics

Usage:
    pip install numpy matplotlib scipy
    python prime_torus_fib_scaling.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy

# ——— Config ———
FIB_DIMS = [1, 2, 3, 5, 8, 13, 21]  # Up to Fibonacci(8)
T_MAX = 30
N_POINTS = 3000
N_SAMP = 200

# ——— Prime generator ———
def generate_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes

# ——— Core torus angle computation ———
def torus_angles(primes, t):
    logs = np.log(primes)
    return (2 * np.pi * t[None, :] / logs[:, None]) % (2 * np.pi)

# ——— Parallel-coordinates plot ———
def plot_parallel_coords(theta, primes):
    d, N = theta.shape
    norm = theta / (2 * np.pi)
    xs = np.arange(d)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(N):
        ax.plot(xs, norm[:, i], alpha=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(primes, rotation=45)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', 'π', '2π'])
    ax.set_title(f"Parallel Coordinates: {len(primes)}D Torus")
    plt.tight_layout()
    plt.show()

# ——— Random 3D projection ———
def plot_random_3d(theta, primes):
    d, N = theta.shape
    if d < 3:
        print(f"  [!] Cannot project {d}D data into 3D — skipping.")
        return
    centered = theta - theta.mean(axis=1, keepdims=True)
    Q, _ = np.linalg.qr(np.random.randn(d, 3))
    proj = Q.T @ centered
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(proj[0], proj[1], proj[2], lw=0.7)
    ax.set_title(f"3D Projection of {len(primes)}D Torus")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.show()

# ——— Composite signal ———
def composite_signal(theta):
    return np.sum(np.cos(theta), axis=0)

def plot_composite_signal(f, t):
    plt.figure(figsize=(6, 3))
    plt.plot(t, f)
    plt.title("Composite Signal: Σ_i cos(θ_i)")
    plt.xlabel("Time t")
    plt.ylabel("f(t)")
    plt.tight_layout()
    plt.show()

# ——— Entropy ———
def compute_entropy(theta, bins=50):
    flat = theta.flatten()
    hist, _ = np.histogram(flat, bins=bins, range=(0, 2 * np.pi), density=True)
    hist += 1e-12
    return entropy(hist)

# ——— Angular speed norms ———
def angular_velocity_norm(theta, t):
    dt = np.diff(t)
    dtheta = np.diff(theta, axis=1)
    speed = np.linalg.norm(dtheta, axis=0) / dt
    return speed

# ——— Main Execution ———
def main():
    t_full = np.linspace(0, T_MAX, N_POINTS)
    t_samp = np.linspace(0, T_MAX, N_SAMP)
    all_primes = generate_primes(max(FIB_DIMS))

    for d in FIB_DIMS:
        primes = all_primes[:d]
        print(f"\n==> Running {d}-dimensional prime torus with primes: {primes}")

        θ_samp = torus_angles(primes, t_samp)
        θ_full = torus_angles(primes, t_full)

        # Parallel coordinates
        plot_parallel_coords(θ_samp, primes)

        # 3D projection (conditionally skipped)
        plot_random_3d(θ_full, primes)

        # Composite signal
        f = composite_signal(θ_full)
        plot_composite_signal(f, t_full)

        # Metrics
        speed = angular_velocity_norm(θ_full, t_full)
        H = compute_entropy(θ_full)

        print(f"  ⟳ Mean angular speed: {np.mean(speed):.3f}")
        print(f"  ⟳ Std  angular speed: {np.std(speed):.3f}")
        print(f"  ⚛︎ Angular entropy  : {H:.4f} nats")

if __name__ == '__main__':
    main()
