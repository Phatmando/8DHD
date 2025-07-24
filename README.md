# Prime Waves Toolkit – Overview

A lightweight launch point for exploring how **prime numbers → triplets → gap sequences → FM waves → 2‑D interference fields**, plus the broader theoretical context (Ω–Φ framework, 8DHD, fractal prime structures).

---

## What you’ll find here

- **`prime_fm.py`** – Python script that:
  - Generates primes, groups them into triplets (X, Y, Z).
  - Converts X/Y gaps into FM sine waves.
  - Builds additive & multiplicative interference heatmaps.
- **`latex2pdf_letter.py`** – Python helper that compiles a hard‑coded LaTeX block (the algorithmic steps) into a clean US‑Letter PDF.
- **`equations.pdf`** – The rendered LaTeX explainer of the math/algorithm pipeline.
- **Link to the website** *(you’ll add this)* – High‑level narrative, visuals, and ongoing updates.
- **“The Ω–Φ Binary Framework: From Prime Waves to Physical Realization” (PDF)** – Deeper theoretical paper tying prime “waves” to Ω (π‑phase flips) and Φ (golden‑ratio scale jumps), embedded in an 8‑dimensional holographic model.

 # Prime-Triplet Field

![2D FM-Sine Interference of X & Y](images/2d_pwave_interference.png)

## Scripts

- [latex2pdf_letter.py](scripts/latex2pdf_letter.py)  
- [prime_fm.py](scripts/prime_fm.py)  

## Download

- [All scripts (ZIP)](scripts.zip)  

---

## Quick start

```bash
# visualize interference fields
python prime_fm.py --triplets 200 --up 20 --save out.png

# render the LaTeX math sheet to PDF
python latex2pdf_letter.py -o equations.pdf

    Edit parameters with CLI flags instead of touching code.

Topics at a glance (very brief)

    Prime Triplet Structure
    Primes are grouped in sequential triplets (… p₁,p₂,p₃ | p₄,p₅,p₆ …). Extracting the 1st, 2nd, 3rd of each triplet gives three “outer strings” (X, Y, Z). Their gaps encode non‑random structure.

    Gap → Frequency Mapping
    Normalize gaps to [0,1], map to a frequency band [f_min, f_max], upsample, integrate → FM sine waves. Two such waves (from X & Y) interfere to form rich 2‑D patterns (sum/product heatmaps).

    Ω–Φ Binary Moves

        Ω: π‑phase inversion (flip/sign change).

        Φ: Golden‑ratio scale dilation (~1.618).
        Together they form a minimal “alphabet of change” for nested, self‑similar layers.

    Six “Prime Waves” & Spectral Hints
    Empirical fits suggest six dominant oscillatory modes in prime distributions—linked to leading Riemann zeta zeros—interpreted as six scalar degrees of freedom on a compact T⁶.

    8DHD Context
    An Eight‑Dimensional Holographic Duality frame: nested toroidal layers with alternating orientation, E₈ lattice embeddings for gauge unification, Π‑twist recursion giving quantized “resonant foyers,” and a geometric take on inflation/Λ.

    Fractal Prime Field
    Recursively re‑tripletizing each outer string yields deeper “inner strings,” building a branching fractal tree of prime-derived sequences with repeating motifs.

Minimal requirements

    Python 3.8+

    numpy, matplotlib (for prime_fm.py)

    A TeX engine: pdflatex or (latex + dvipdf) for latex2pdf_letter.py


Coming soon:

    More images/audio (e.g., turn FM waves into sound and see the soundwaves in action in 3d medium)

    Add small tests (prime generation, array shapes).

    I did not find this prime triplet sequence and credit is due to the founder that the prime scalar field website and reddit community.
    I am merely drawing conculsions on a shape that can support the structure of the primes oscilation in real space and time.
    More informations can be found at the following sites

    (https://theprimescalarfield.com/)

Questions, issues, or ideas? Open an issue or reach out!
