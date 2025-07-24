import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

# =============================================================================
# Module: prime_torus_8dhd.py
# =============================================================================

def generate_primes(n):
    """
    Generate the first n prime numbers via trial division.
    """
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes

def build_time_array(primes, T=50.0, N=2000):
    """
    Build a time grid [0, T] of N points, including prime integers <= T.
    """
    dense = np.linspace(0, T, N)
    prime_ts = [p for p in primes if p <= T]
    t = np.unique(np.concatenate((dense, prime_ts)))
    return t

def compute_prime_angles(primes, t):
    """
    Compute θ_{p_i}(t) = (2π * t / ln(p_i)) mod 2π for each prime p_i.
    """
    thetas = np.zeros((len(t), len(primes)))
    for i, p in enumerate(primes):
        thetas[:, i] = (2 * np.pi * t / np.log(p)) % (2 * np.pi)
    return thetas

def plot_parallel_coordinates(thetas, primes, sample_cnt=6):
    """
    Plot θ/(2π) vs. prime index for harmonic crossings.
    """
    norm = thetas / (2 * np.pi)
    idxs = np.linspace(0, len(norm) - 1, sample_cnt, dtype=int)
    plt.figure(figsize=(6, 4))
    for idx in idxs:
        plt.plot(primes, norm[idx], alpha=0.6)
    plt.xlabel("Prime p_i")
    plt.ylabel("θ/(2π)")
    plt.title("Parallel Coordinates of Torus Flow")
    plt.show()

def project_to_3d(thetas):
    """
    Project high-dimensional flow into 3D for visualization.
    """
    centered = thetas - thetas.mean(axis=0)
    G = np.random.randn(centered.shape[1], 3)
    Q, _ = np.linalg.qr(G)
    return centered.dot(Q)

def compute_composite_signal(thetas):
    """
    Composite signal: f(t) = Σ_i cos(θ_i(t)).
    """
    return np.sum(np.cos(thetas), axis=1)

def find_local_minima(f, order=10):
    """
    Detect local minima using a sliding-window comparator.
    """
    return argrelextrema(f, np.less, order=order)[0]

def sample_at_prime_times(primes, thetas, t):
    """
    Extract torus states at integer prime times t = p.
    """
    idx_map = {val: i for i, val in enumerate(t)}
    return np.vstack([thetas[idx_map[p]] for p in primes if p in idx_map])

def pi_twist(thetas, primes):
    """
    Apply π-twist: θ_i -> (θ_i + π + 1/ln(p_i)) mod 2π.
    """
    twist = np.zeros_like(thetas)
    for i, p in enumerate(primes):
        twist[:, i] = (thetas[:, i] + np.pi + 1 / np.log(p)) % (2 * np.pi)
    return twist

def find_recurrence_times(thetas, twisted, eps=1.0):
    """
    Identify times where twisted trajectory returns near origin.
    """
    diffs = np.linalg.norm((twisted - thetas[0]) % (2 * np.pi), axis=1)
    return np.where(diffs < eps)[0]

def symbolic_encoding(thetas, M=12):
    """
    Bin angle values into M discrete symbols.
    """
    bins = np.linspace(0, 2 * np.pi, M + 1)
    s = np.digitize(thetas, bins) - 1
    s[s == M] = M - 1
    return s

def compute_kde_density(thetas, j, k, grid=100):
    """
    Kernel density estimation for angle coordinates j and k.
    """
    data = np.vstack([thetas[:, j], thetas[:, k]])
    kde = gaussian_kde(data)
    xi = np.linspace(0, 2 * np.pi, grid)
    yi = np.linspace(0, 2 * np.pi, grid)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(grid, grid)
    return X, Y, Z

# =============================================================================
# Main Execution: run pipeline on d = 3, 5, 8
# =============================================================================

for d in (3, 5, 8):
    print(f"\n### Running pipeline on T^{d} torus ###")
    primes = generate_primes(d)
    t = build_time_array(primes)
    thetas = compute_prime_angles(primes, t)

    plot_parallel_coordinates(thetas, primes)
    Y3 = project_to_3d(thetas)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Y3[:, 0], Y3[:, 1], Y3[:, 2], lw=0.5)
    ax.set_title(f"3D Projection of T^{d} Trajectory")
    plt.show()

    f = compute_composite_signal(thetas)
    minima = find_local_minima(f)
    print("Minima times:", t[minima][:5], "...", f"[total {len(minima)} minima]")

    plt.figure(figsize=(5, 3))
    plt.plot(t, f, label='f(t)')
    plt.scatter(t[minima], f[minima], color='red', s=10, label='minima')
    plt.title("Composite Harmonic Signal")
    plt.legend()
    plt.show()

    samples = sample_at_prime_times(primes, thetas, t)
    print("Prime-time samples shape:", samples.shape)

    twisted = pi_twist(thetas, primes)
    rec = find_recurrence_times(thetas, twisted)
    print("Recurrence count (<1 rad):", len(rec))

    sym = symbolic_encoding(thetas)
    print("Symbolic encoding (first 3 rows):\n", sym[:3])

    X, Y, Z = compute_kde_density(thetas, 0, 1)
    plt.figure(figsize=(4, 4))
    plt.contourf(X, Y, Z, levels=15)
    plt.title("2D Subtorus KDE (axes 0,1)")
    plt.xlabel("θ_0")
    plt.ylabel("θ_1")
    plt.show()
