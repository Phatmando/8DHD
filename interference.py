import numpy as np

import matplotlib.pyplot as plt

from math import isqrt, log



# ─────────── Utility Functions ───────────



def first_n_primes(n):

    """Return the first n primes using a simple sieve."""

    # Overestimate upper bound p_n ≈ n (log n + log log n)

    B = int(n*(log(n) + log(log(n))) + 100)

    sieve = np.ones(B+1, bool)

    sieve[:2] = False

    for p in range(2, isqrt(B)+1):

        if sieve[p]:

            sieve[p*p : B+1 : p] = False

    primes = np.nonzero(sieve)[0]

    return primes[:n].tolist()



def tripletize(seq):

    """Split seq into (X,Y,Z) outer‐string lists."""

    xs, ys, zs = [], [], []

    for i in range(len(seq)//3):

        a, b, c = seq[3*i : 3*i+3]

        xs.append(a); ys.append(b); zs.append(c)

    return xs, ys, zs



def compute_gaps(seq):

    """Return successive differences of seq."""

    return np.diff(seq).astype(float)



def fm_wave_from_gaps(gaps, n_trips, up=20, fmin=0.02, fmax=0.08):

    """

    Build an FM sine from the first n_trips gaps:

      • normalize gaps→[0,1]

      • map to inst. freq fmin→fmax

      • upsample by up

      • integrate to get phase and return sin(phase)

    """

    if len(gaps) < n_trips:

        raise ValueError(f"Need {n_trips} gaps, got {len(gaps)}")

    g = gaps[:n_trips]

    gmin, gmax = g.min(), g.max()

    norm = (g - gmin)/(gmax - gmin)

    f_inst = fmin + norm*(fmax - fmin)

    # upsample in “triplet‐units”

    t_coarse = np.arange(n_trips)

    t_fine   = np.linspace(0, n_trips-1, n_trips*up)

    f_fine   = np.interp(t_fine, t_coarse, f_inst)

    phase    = 2*np.pi * np.cumsum(f_fine)/up

    return t_fine, np.sin(phase)



# ─────────── Main ───────────



if __name__ == "__main__":

    # ◉ User parameters ◉

    N_TRIPLETS = 200    # how many gaps (& thus triplets) to visualize

    UPSAMPLE   = 20     # 20 waveform‐samples per triplet

    #─────────────────────────────────────



    # 1) Generate enough primes for N_TRIPLETS gaps

    needed_primes = 3*(N_TRIPLETS+1)

    primes = first_n_primes(needed_primes)

    print(f"Generated {len(primes)} primes ⇒ {len(primes)//3} triplets")



    # 2) Build outer‐strings X and Y

    X, Y, Z = tripletize(primes)

    gaps_X = compute_gaps(X)

    gaps_Y = compute_gaps(Y)



    # 3) Turn those gaps into FM‐sine waves

    t, wave_X = fm_wave_from_gaps(gaps_X, N_TRIPLETS, UPSAMPLE)

    _, wave_Y = fm_wave_from_gaps(gaps_Y, N_TRIPLETS, UPSAMPLE)



    # 4a) Additive interference field

    field_add = np.add.outer(wave_X, wave_Y)   # shape (T,T)



    # 4b) Multiplicative “beat” field

    field_mul = np.outer(wave_X, wave_Y)       # same shape



    # 5) Plot both

    plt.figure(figsize=(12,5))



    plt.subplot(1,2,1)

    plt.imshow(field_add,

               origin='lower',

               cmap='inferno',

               aspect='auto',

               extent=[0, N_TRIPLETS, 0, N_TRIPLETS])

    plt.colorbar(label='X+Y amplitude')

    plt.title('Additive Interference\nX+Y')

    plt.xlabel('Y triplet idx')

    plt.ylabel('X triplet idx')



    plt.subplot(1,2,2)

    plt.imshow(field_mul,

               origin='lower',

               cmap='inferno',

               aspect='auto',

               extent=[0, N_TRIPLETS, 0, N_TRIPLETS])

    plt.colorbar(label='X·Y amplitude')

    plt.title('Multiplicative Interference\nX×Y (Moiré)')

    plt.xlabel('Y triplet idx')

    plt.ylabel('X triplet idx')



    plt.suptitle(f'2D Interference Patterns of X & Y (first {N_TRIPLETS} triplets)', y=0.95)

    plt.tight_layout(rect=[0,0,1,0.93])

    plt.show()
