#!/usr/bin/env python3
"""
prime_fm.py
Turn prime triplet gaps into FM waves and 2D interference heatmaps.

Usage:
    python prime_fm.py --triplets 200 --up 20 --save out.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import isqrt, log

# ─────────── Prime utilities ───────────
def first_n_primes(n: int):
    if n < 6:
        # tiny fallback so log(log(n)) doesn't go negative
        small = [2,3,5,7,11,13][:n]
        return small
    B = int(n * (log(n) + log(log(n))) + 100)
    sieve = np.ones(B+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, isqrt(B)+1):
        if sieve[p]:
            sieve[p*p:B+1:p] = False
    return np.nonzero(sieve)[0][:n].tolist()

def tripletize(seq):
    xs, ys, zs = [], [], []
    for i in range(len(seq)//3):
        a,b,c = seq[3*i:3*i+3]
        xs.append(a); ys.append(b); zs.append(c)
    return xs, ys, zs

def compute_gaps(seq):
    return np.diff(seq).astype(float)

# ─────────── Signal builder ───────────
def fm_wave_from_gaps(gaps, n_trips, up=20, fmin=0.02, fmax=0.08):
    if len(gaps) < n_trips:
        raise ValueError(f"Need {n_trips} gaps, got {len(gaps)}")
    g = gaps[:n_trips]
    gmin, gmax = float(g.min()), float(g.max())
    span = gmax - gmin
    norm = np.zeros_like(g) if span == 0 else (g - gmin)/span
    f_inst = fmin + norm*(fmax - fmin)

    t_coarse = np.arange(n_trips)
    t_fine   = np.linspace(0, n_trips-1, n_trips*up)
    f_fine   = np.interp(t_fine, t_coarse, f_inst)
    phase    = 2*np.pi * np.cumsum(f_fine)/up
    return t_fine, np.sin(phase)

# ─────────── Main ───────────
def main():
    ap = argparse.ArgumentParser(description="Prime FM interference plots.")
    ap.add_argument("--triplets", type=int, default=200, help="Number of gaps/triplets to use")
    ap.add_argument("--up", type=int, default=20, help="Upsample factor")
    ap.add_argument("--fmin", type=float, default=0.02)
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--save", type=str, default=None, help="Path to save PNG (if omitted, just show)")
    args = ap.parse_args()

    needed_primes = 3*(args.triplets + 1)
    primes = first_n_primes(needed_primes)
    print(f"Generated {len(primes)} primes ⇒ {len(primes)//3} triplets")

    X, Y, Z = tripletize(primes)
    gaps_X = compute_gaps(X)
    gaps_Y = compute_gaps(Y)

    _, wave_X = fm_wave_from_gaps(gaps_X, args.triplets, args.up, args.fmin, args.fmax)
    _, wave_Y = fm_wave_from_gaps(gaps_Y, args.triplets, args.up, args.fmin, args.fmax)

    field_add = np.add.outer(wave_X, wave_Y)
    field_mul = np.outer(wave_X, wave_Y)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    im1 = plt.imshow(field_add, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(im1, label='X+Y amplitude')
    plt.title('Additive Interference (X+Y)')
    plt.xlabel('Y index'); plt.ylabel('X index')

    plt.subplot(1,2,2)
    im2 = plt.imshow(field_mul, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(im2, label='X·Y amplitude')
    plt.title('Multiplicative Interference (X×Y)')
    plt.xlabel('Y index'); plt.ylabel('X index')

    plt.suptitle(f'Interference of X & Y (first {args.triplets} triplets)', y=0.97)
    plt.tight_layout(rect=[0,0,1,0.94])

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
