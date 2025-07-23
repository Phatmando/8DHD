#!/usr/bin/env python3
"""
latex2pdf_letter.py
-------------------
Render a LaTeX math block to a nicely formatted PDF on US Letter (8.5x11) paper.

- Uses article class, 1" margins, portrait orientation.
- No cropping step (full, standard page).
- Falls back to latex+dvipdf if pdflatex is missing.

Usage:
    python latex2pdf_letter.py -o equations.pdf
    python latex2pdf_letter.py -o eq.pdf --keep-temp
"""

import argparse
from pathlib import Path
import shutil, subprocess, tempfile, sys

# ---------- LaTeX content ----------
LATEX_BODY = r"""
\section*{1. Prime Generation (Sieve of Eratosthenes)}

We wish to generate the first \(N_p\) primes.  Choose an upper bound
\[
B \;\ge\; N_p\bigl(\ln N_p + \ln\ln N_p\bigr)\,.
\]
Initialize a boolean array \(\mathrm{isPrime}[0\ldots B]\) with
\(\mathrm{isPrime}[0]=\mathrm{isPrime}[1]=\mathsf{false}\) and
\(\mathrm{isPrime}[i]=\mathsf{true}\) for \(i\ge2\).  Then:

\begin{align*}
\text{For }i&=2,\dots,\lfloor\sqrt B\rfloor,\\
\text{if }\mathrm{isPrime}[i]\text{ then mark }
\mathrm{isPrime}[i^2],\,\mathrm{isPrime}[i^2+i],\,\mathrm{isPrime}[i^2+2i],\dots=\mathsf{false}.
\end{align*}

The first \(N_p\) remaining true indices are
\(\{p_1,p_2,\dots,p_{N_p}\}\).

\section*{2. Extract Outermost Triplets}

Let
\[
T \;=\;\Bigl\lfloor\tfrac{N_p}{3}\Bigr\rfloor
\]
be the number of non‐overlapping triplets.  For
\(k=0,1,\dots,T-1\) define
\[
X_k = p_{3k+1},\quad
Y_k = p_{3k+2},\quad
Z_k = p_{3k+3}.
\]

\section*{3. Gap Sequences}

For each string \(S\in\{X,Y,Z\}\), form
\[
\Delta^S_k = S_{k+1} - S_k,\quad k=0,\dots,T-2.
\]

\section*{4. Normalize Gaps to Instantaneous Frequencies}

Set
\(\Delta_{\min}=\min_k\Delta^S_k\),
\(\Delta_{\max}=\max_k\Delta^S_k\), and
\[
\widetilde\Delta_k
=\frac{\Delta^S_k - \Delta_{\min}}{\Delta_{\max}-\Delta_{\min}}
\in[0,1].
\]
Choose bounds \(f_{\min},f_{\max}\), then
\[
f_k
= f_{\min}
+ \widetilde\Delta_k\,(f_{\max}-f_{\min}),
\quad k=0,\dots,N-1.
\]

\section*{5. Upsample and Interpolate}

Pick integer \(u\).  Define
\[
t_i = \frac{i}{u},\quad i=0,1,\dots,u\,N-1,
\]
and interpolate
\(\{(k,f_k)\}\) to obtain \(f(t_i)\) by linear interpolation.

\section*{6. Phase Accumulation and Waveform}

Initialize \(\Phi_0=0\).  Then
\[
\Phi_i = \Phi_{i-1} + 2\pi\,\frac{f(t_i)}{u},
\qquad
s_i = \sin(\Phi_i),
\quad i=0,\dots,u\,N-1.
\]

\section*{7. Two‐Dimensional Interference Fields}

Given two waves \(\{s_i^X\}\) and \(\{s_j^Y\}\), define
\[
A_{ij} = s_i^X + s_j^Y,
\quad
M_{ij} = s_i^X \times s_j^Y,
\quad i,j=0,\dots,u\,N-1.
\]
Plot \(A\) and \(M\) as heatmaps over the grid \([0,uN-1]^2\).

\section*{Parameter Summary}

\begin{itemize}
  \item \(N_p\): number of primes
  \item \(T=\lfloor N_p/3\rfloor\): number of triplets
  \item \(N\le T-1\): gaps used
  \item \(u\): upsampling factor
  \item \(f_{\min},f_{\max}\): frequency bounds
\end{itemize}
"""

TEMPLATE = r"""
\documentclass[letterpaper,12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{lmodern}
\setlength{\parindent}{0pt}
\begin{document}
%s
\end{document}
"""

# ---------- helpers ----------
def run(cmd, cwd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def compile_to_pdf(tex_src: str, out_pdf: Path, keep_temp=False):
    have_pdflatex = shutil.which("pdflatex")
    have_latex    = shutil.which("latex")
    have_dvipdf   = shutil.which("dvipdf")

    if not (have_pdflatex or (have_latex and have_dvipdf)):
        raise SystemExit("Need pdflatex OR (latex + dvipdf). Install a full TeX distribution.")

    tmpdir = tempfile.TemporaryDirectory()
    tmp = Path(tmpdir.name)
    (tmp / "eq.tex").write_text(tex_src, encoding="utf8")

    try:
        if have_pdflatex:
            run(["pdflatex", "-interaction=nonstopmode", "eq.tex"], cwd=tmpdir.name)
        else:
            run(["latex", "-interaction=nonstopmode", "eq.tex"], cwd=tmpdir.name)
            run(["dvipdf", "eq.dvi", "eq.pdf"], cwd=tmpdir.name)

        pdf_path = tmp / "eq.pdf"
        if not pdf_path.exists():
            raise RuntimeError("PDF was not created.")
        pdf_path.replace(out_pdf)
        print(f"Saved {out_pdf}")
    finally:
        if keep_temp:
            print(f"Temp files kept at: {tmpdir.name}")
        else:
            tmpdir.cleanup()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Render LaTeX block to a letter-sized PDF.")
    ap.add_argument("-o", "--output", default="equations.pdf", help="output PDF file")
    ap.add_argument("--keep-temp", action="store_true", help="do not delete temp build folder")
    args = ap.parse_args()

    tex_src = TEMPLATE % LATEX_BODY
    compile_to_pdf(tex_src, Path(args.output).resolve(), keep_temp=args.keep_temp)

if __name__ == "__main__":
    main()
