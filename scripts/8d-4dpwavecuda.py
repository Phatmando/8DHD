#!/usr/bin/env python3
import cupy as cp
import matplotlib.pyplot as plt
from math import isqrt, log
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from matplotlib.animation import FuncAnimation

# ─── Prime utilities ────────────────────────────────────────────────────
def first_n_primes(n: int):
    """Return the first n primes (treating 1 as the first “prime” if n<6)."""
    if n < 6:
        return [2, 3, 5, 7, 11, 13][:n]
    # Estimate upper bound with n*(log n + log log n)
    B = int(n * (log(n) + log(log(n))) + 100)
    sieve = cp.ones(B + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, isqrt(B) + 1):
        if sieve[p]:
            sieve[p*p : B+1 : p] = False
    # Return Python list of the first n primes
    return cp.nonzero(sieve)[0][:n].tolist()

def tripletize(seq):
    """Split a Python list into X,Y,Z arrays of triplets on the GPU."""
    xs, ys, zs = [], [], []
    for i in range(len(seq) // 3):
        a, b, c = seq[3*i : 3*i + 3]
        xs.append(a); ys.append(b); zs.append(c)
    return cp.array(xs), cp.array(ys), cp.array(zs)

def compute_gaps(arr):
    """Compute successive differences on a GPU array."""
    return cp.diff(arr).astype(cp.float32)

def fm_phase_from_gaps(gaps, n_trips, up=20, fmin=0.02, fmax=0.08):
    """Generate the instantaneous phase for an FM wave from prime gaps."""
    g = gaps[:n_trips]
    gmin = float(cp.min(g)); gmax = float(cp.max(g))
    if gmax > gmin:
        norm = (g - gmin) / (gmax - gmin)
    else:
        norm = cp.zeros_like(g)
    f_inst = fmin + norm * (fmax - fmin)

    t_coarse = cp.arange(n_trips)
    t_fine   = cp.linspace(0, n_trips - 1, n_trips * up)
    f_fine   = cp.interp(t_fine, t_coarse, f_inst)
    phase    = 2 * cp.pi * cp.cumsum(f_fine) / up
    return t_fine, phase

# ─── Main script ────────────────────────────────────────────────────────
def main():
    # Parameters
    n_trips = 200        # Number of prime triplets
    up      = 20         # Upsampling factor for smooth wave
    R, r0, amp = 2.0, 0.5, 0.2   # Torus (major radius, base minor radius, modulation amplitude)
    NODE_THRESH = 0.03          # Threshold to mark nodes (zero crossings)

    # 1) Generate FM phase on GPU
    primes = first_n_primes(3 * (n_trips + 1))
    X, Y, Z = tripletize(primes)
    gaps_X  = compute_gaps(X)
    t_fine, phase = fm_phase_from_gaps(gaps_X, n_trips, up)

    # 2) Prepare torus parameter grid on GPU
    nθ = len(t_fine)
    nφ = 60
    θ  = cp.linspace(0, 2 * cp.pi, nθ)
    φ  = cp.linspace(0, 2 * cp.pi, nφ)
    Θ, Φ = cp.meshgrid(θ, φ)

    # 3) Set up Matplotlib figure
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    # 4) Animation update function
    def update(frame):
        ax.clear()
        shift = 2 * cp.pi * (frame / 100)
        wave  = cp.sin(phase + shift)      # GPU: instantaneous amplitude
        W     = cp.tile(wave, (nφ, 1))     # Broadcast to φ×θ grid
        r     = r0 + amp * W               # Modulated minor radius

        # Compute 3D coordinates on GPU
        X3 = (R + r * cp.cos(Φ)) * cp.cos(Θ)
        Y3 = (R + r * cp.cos(Φ)) * cp.sin(Θ)
        Z3 =               r * cp.sin(Φ)

        # Normalize for colormap
        C  = (W - cp.min(W)) / (cp.max(W) - cp.min(W))

        # Bring small arrays to CPU for plotting
        X3_cpu = cp.asnumpy(X3)
        Y3_cpu = cp.asnumpy(Y3)
        Z3_cpu = cp.asnumpy(Z3)
        C_cpu  = cp.asnumpy(C)

        # Render the torus surface with colormap
        ax.plot_surface(
            X3_cpu, Y3_cpu, Z3_cpu,
            facecolors=plt.cm.coolwarm(C_cpu),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False
        )

        # Overlay nodal points (|wave| < NODE_THRESH)
        nodes = (cp.abs(W) < NODE_THRESH)
        xs = cp.asnumpy(X3[nodes])
        ys = cp.asnumpy(Y3[nodes])
        zs = cp.asnumpy(Z3[nodes])
        ax.scatter(xs, ys, zs, c='k', s=1, alpha=0.8)

        ax.set_title(f"Frame {frame}: Prime-FM Resonance", y=0.9)

    # 5) Run the animation
    anim = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
