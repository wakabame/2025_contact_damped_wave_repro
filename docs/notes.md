# Reproduction notes — arXiv:2412.06185, Section 6

A record of what was learned while implementing and validating the reproduction of the
paper's two examples and our own Example 3. Every number comes from the output of
`uv run python scripts/run_examples.py` / `scripts/convergence_study.py` in this repository.

---

## 1. The scheme, reorganized for implementation

With the discrete Laplacian $(D u)_j=(u_{j+1}-2u_j+u_{j-1})/\Delta x^2$, the finite-difference
equation of §6 of the paper turns into a **time-independent symmetric positive-definite
tridiagonal system** per step:

$$
\big[I-(\alpha\Delta t+\Delta t^2)D\big]\eta^{i+1}
= 2\eta^i-\eta^{i-1}-\alpha\Delta t\,D\eta^i+\Delta t^2P^i,\qquad
P^i_j=\frac1\varepsilon\mathbf 1_{\{\eta^i_j<0\}}\Big(\frac{\eta^i_j-\eta^{i-1}_j}{\Delta t}\Big)^{-}.
$$

At the paper's parameters $(\alpha\Delta t+\Delta t^2)/\Delta x^2=51$, i.e. the matrix is
$\mathrm{tridiag}(-51,103,-51)$. It is factorized once with `cholesky_banded` and each step
is an $O(N)$ `cho_solve_banded`. At the paper's resolution ($N=5000$) Example 1 runs in
0.30 s, Example 2 in 0.48 s, Example 3 in 0.66 s.

### 1.1 An exact discrete energy identity

Testing the scheme with $v^{i+1/2}=(\eta^{i+1}-\eta^i)/\Delta t$ and summing (summation by
parts is exact because $v=0$ at the endpoints) gives, for
$E^i=\frac{\Delta x}2\sum_j(v^{i-1/2}_j)^2+\frac{\Delta x}2\sum_j(\delta_x\eta^i_j)^2$,

$$
E^{i+1}-E^i=-\Delta t\,[\,Q_{\mathrm{visc}}+Q_{\mathrm{num}}+Q_{\mathrm{con}}\,],
$$
$$
Q_{\mathrm{visc}}=\alpha\Delta x\!\sum_j(\delta_xv^{i+1/2}_j)^2,\quad
Q_{\mathrm{num}}=\frac{\Delta x}{2\Delta t}\!\sum_j(v^{i+1/2}_j-v^{i-1/2}_j)^2+\frac{\Delta t}2\Delta x\!\sum_j(\delta_xv^{i+1/2}_j)^2,
$$
$$
Q_{\mathrm{con}}=-\Delta x\sum_jP^i_jv^{i+1/2}_j .
$$

$Q_{\mathrm{visc}}$ is the discrete counterpart of $\alpha\int|\partial_{tx}\eta|^2$ in (1.4),
$Q_{\mathrm{con}}$ of $\int D_{\mathrm{con}}$, and $Q_{\mathrm{num}}$ is the numerical
dissipation of the scheme itself. $Q_{\mathrm{num}}$ is $O(\Delta t)$ for smooth solutions at
fixed $\varepsilon$, but it does not vanish uniformly through impacts when
$\Delta t/\varepsilon$ is held fixed: **even at the paper's resolution it carries 14–18% of
the total energy loss** (Ex. 1: 18%, Ex. 2: 14%). The solver records all three rates at
every step, so the budget **closes to machine precision** (drift $\sim10^{-10}$, the round-off
accumulated over 1500–2500 steps).

> **Initial implementation mistake**: evaluating $\alpha\int|\delta_xv|^2$ with the backward
> velocity $v^{i-1/2}$ and integrating with the trapezoidal rule gave a budget drift of 17.8
> (a cumulative viscous dissipation of 23293 against an actual dissipation of 1300). Switching
> to the identity above resolved it.

### 1.2 Stability condition of the explicit penalty (not stated in the paper)

While penetrating ($\eta<0,\ v<0$) the velocity update is close to
$v\leftarrow(1-\Delta t/\varepsilon)v$, so

- $\Delta t/\varepsilon<1$: monotone decay (no bounce; consistent with assumption (A4) of
  the theory)
- $1\le\Delta t/\varepsilon<2$: decays but oscillates (a spurious bounce). Linearly stable,
  yet already at $\Delta t/\varepsilon=1$ the contact set degrades badly (table in §4)
- $\Delta t/\varepsilon\ge2$: amplifies; the energy grows and the run blows up

Numerical check (Example 1, $\Delta x=\Delta t=1/500$, $T=0.1$):

| $\Delta t/\varepsilon$ | 0.1 | 0.2 | 0.4 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| $E(T)$ | 47.1 | 32.4 | 24.6 | 23.0 | 19.8 | **432.3** |
| cumulative contact work | 1064 | 1013 | 868 | 784 | 214 | **−2372** |

The paper's setting, $\Delta t/\varepsilon=0.4$, is safely monotone. **Because of this
constraint, $\varepsilon\to0$ and $\Delta t\to0$ cannot be taken independently** (refining
$\varepsilon$ requires refining $\Delta t$ with it).

### 1.3 Penetration depth

While penetrating, the ODE is $v'=-v/\varepsilon$, so $\max(0,-\eta)\approx|v_0|\varepsilon$.
At the paper's parameters $|v_0|\varepsilon=50\times5\times10^{-4}=0.025$; measured
$2.56\times10^{-2}$ (Ex. 1) / $2.49\times10^{-2}$ (Ex. 2). Consistent with the theoretical
$\eta\ge-C\varepsilon$ (with $C\approx|v_0|$).

---

## 2. Inconsistencies in the paper's description and what we adopted

### (b) Example 2's initial data — decisive

Using the printed formulas `x`, `sin(π(x−0.2)/0.3)`, `2−x` **verbatim** gives:

| | figure-consistent (default) | printed verbatim |
|---|---|---|
| max difference between adjacent nodes | $2.09\times10^{-3}$ (= Lipschitz constant $\pi/0.3$ × $\Delta x$) | **1.202** (jump at $x=0.8$) |
| $\eta^0(0)$ | 1 | **0** (violates the assumption $\eta_0\ge c>0$ of Thm 2.1) |
| $\min\eta^0$ | 1 (local minimum at $x=0.65$) | **−1** (at $x=0.65$; negative on $(0.5,0.8)$, 30% of the string starts below the obstacle) |
| elastic energy | 21.4 | **3729.0** (from the interior discontinuities) |
| added by endpoint clamping | 0 | **+2499.0** (forcing $\eta^0(0)=0\to h=1$) |
| $E(0)$ (= elastic + kinetic 749.8) | 771.2 | **6977.8** (9× too large) |
| first contact time | 0.0256 | **0.0000** ($\eta^0<0$ on $(0.5,0.8)$, in contact from the start) |

Breakdown of $E(0)$ for the verbatim formula: 3729 from the interior discontinuities, 2499
from forcibly clamping the mismatch with boundary condition (1.3) (a numerical artifact that
diverges as $\Delta x\to0$), kinetic 749.8. Both contributions dwarf the figure variant's
elastic energy of 21.4.

Fig. 4(a) shows a continuous profile with endpoints at 1, value 2 at $x=0.2,0.5,0.8$,
maximum 3 (at $x=0.35$), local minimum 1 (at $x=0.65$). The default figure variant satisfies
all of this (verified in `tests/test_initial_data.py`). **The printed formula cannot produce
Fig. 4.** It can be run with `--initial paper-literal` (`results/ex2_paper_literal/`).

### (a) Boundary condition

The paper prints $\eta^i_0=\eta^i_N=0$, but the PDE (1.3) prescribes $\eta=h>0$ and
Figs. 2/4 show the endpoints at 1. We adopt $h=1$.

The initial data are defined in the form $\eta^0=h+\text{shape}(x)$ (with
$\text{shape}(0)=\text{shape}(l)=0$), so they are compatible with (1.3) for any $h$.
Dependence on $h$:

| $h$ | 0.5 | 1.0 (paper) | 1.5 | 2.0 |
|---|---|---|---|---|
| $E(0)$ | 1311.43 | 1311.43 | 1311.43 | 1311.43 |
| first contact $t_{\min}$ | 0.0112 | 0.0236 | 0.0346 | 0.0436 |
| contact vanishing $t_{\max}$ | 0.3000 | 0.2990 | 0.1984 | 0.2368 |
| contact area | 0.1702 | 0.1542 | 0.1128 | 0.1223 |

$E(0)$ is independent of $h$ because the elastic energy depends only on $\partial_x\eta$.
$t_{\min}$ is monotone in $h$ (a longer fall), but $t_{\max}$ is **non-monotone**. Since the
paper's $h=1$ is pinned down by the endpoint values in Figs. 2/4, this cannot explain the
discrepancy of §3.

> **Caution (an error in the first version)**: the initial data originally hard-coded
> endpoint height 1 regardless of $h$, so for $h\ne1$ the first cell carried a jump of size
> $|1-h|$ and $\sim(1-h)^2/\Delta x$ of spurious elastic energy (divergent as
> $\Delta x\to0$). The conclusion "$h$ has almost no influence" obtained in that state was
> wrong. `solve()` now raises a `RuntimeWarning` when the endpoints disagree with $h$.

### (d) Initial step

Both $\eta^{-1}=\eta^0-\Delta t v^0$ (default) and $\eta^1=\eta^0+\Delta t v^0$ are
implemented. The energy budget closes to machine precision either way (for the latter from
$i\ge1$ on).

### (e) Criterion for the contact set

The difference between $\{\eta<0\}$ and $\{\eta\le\text{tol}\}$ is **below 1% in area**
(1.2% for Example 1 even at tol $=10^{-3}$) — less sensitive than we had feared when
planning. The default is $\{\eta<0\}$ (the **penetration set**; it contains the support of
the penalty force but does not equal it — the force also requires $v<0$, and about 85% of
penetrating nodes have $v\ge0$ and carry zero force). The tolerance can be changed with
`--contact-tol`.

### Other (minor)

- **(c) Index conventions**: the paper's "$0\le i\le l/\Delta x$, $1\le j\le T/\Delta t$" is
  the opposite of how the formulas use the indices. Reading **superscript $i$ = time,
  subscript $j$ = space** makes everything consistent.
- **(f) Caption of Fig. 4**: says "example 1" but is a typo for example 2; treated as
  example 2.

---

## 3. Reproduction results

### Example 1 (Fig. 2, 3)

| item | paper | this reproduction |
|---|---|---|
| first contact | ≈0.02 | 0.0236 |
| $x$-range of the contact set (at its widest) | ≈[0.05, 0.95] | [0.032, 0.968] |
| symmetry | symmetric about $x=0.5$ | symmetric to $\le1.7\times10^{-12}$ (round-off) |
| ripples on the left edge | 10 | 10 ✓ |
| contact vanishing time | **≈0.26** | **0.299** ← discrepancy |
| range of the velocity field | $[-60,10]$ | equivalent (positive band along the detachment front) ✓ |
| contact interval at $t=0.2$ | ≈[0.3,0.7] | ≈[0.28,0.75] |

All six snapshots ($t=0,0.02,0.04,0.06,0.2,0.3$) match Fig. 2 visually.

### Example 2 (Fig. 4, 5)

| item | paper Fig. 5 | this reproduction |
|---|---|---|
| number of connected components | **2** | **2** ✓ |
| large component | $t\in[\sim0.02,\sim0.26]$, $x\in[\sim0.1,\sim0.6]$ | $t\in[0.026,0.261]$, $x\in[0.032,0.594]$ ✓ |
| small component | $t\in[\sim0.25,\sim0.37]$, $x\in[\sim0.65,\sim0.85]$ | $t\in[0.243,0.381]$, $x\in[0.693,0.860]$ ✓ |
| tip of the large component | $x\approx0.35$–$0.4$ | $x\approx0.35$ ✓ |
| range of the velocity field | $[-50,40]$ | equivalent ✓ |

All six snapshots match Fig. 4 visually (the two contact zones at $t=0.04$ with the right
hump at 1.75, the $[0.1,0.6]$ contact at $t=0.08$, and so on).

### The only discrepancy: Example 1's contact vanishing time (0.26 vs 0.30)

- **Converged** with respect to the grid and $\varepsilon$ (still 0.300 at
  $\Delta x=10^{-4}$). Not a numerical error.
- Depends **non-monotonically** on both $h$ and $\alpha$ ($h$: table above; $\alpha$:
  0.005→0.177, 0.01→0.299, 0.02→0.300, 0.05→0.184). The paper states $h=1$ and
  $\alpha=0.01$ explicitly, so a parameter mismatch cannot explain it.
- The paper's own Fig. 2(f) ($t=0.3$) shows $\eta$ apparently stuck at 0 on
  $x\in[0.42,0.58]$, which is **inconsistent within the paper itself** with Fig. 3's
  "vanishes at $t\approx0.26$". At print resolution $\eta\in(0,0.01)$ cannot be
  distinguished from $\eta=0$, so the difference plausibly lies in the treatment of the
  extremely shallow region just before detachment (in our run $\min\eta=-2.3\times10^{-3}$
  at $t=0.27$).
- Most likely caused by unstated implementation details (initial step, treatment near the
  boundary, ...); **we did not tune parameters to match**.

---

## 4. Convergence tests (which the paper says it performed)

`results/convergence/convergence.txt`. This is a **self-convergence** check in the paper's
own setting: the grid sweep refines $\Delta x=\Delta t$ **jointly**, and the $\varepsilon$
sweep runs at a fixed grid (the finest member of each sweep is the reference). It does not
separate the time, space and penalization errors — that would need independent
$\Delta x$/$\Delta t$ sweeps and an $\varepsilon$ refinement with
$\Delta t/\varepsilon\to0$, which the explicit-penalty constraint $\Delta t<\varepsilon$
couples together (§1.2).

Example 1, $\varepsilon=5\times10^{-4}$ fixed:

| $\Delta x=\Delta t$ | $\Delta t/\varepsilon$ | contact area | $\|\eta(T)-\eta_{\mathrm{ref}}\|_\infty$ | note |
|---|---|---|---|---|
| $10^{-3}$ | 2.00 | 0.00099 | 8.50 | **unstable** (amplifies) |
| $5\times10^{-4}$ | 1.00 | 0.0494 | $5.5\times10^{-2}$ | **non-monotone** (bounces) |
| $4\times10^{-4}$ | 0.80 | 0.0861 | $2.6\times10^{-2}$ | |
| $2\times10^{-4}$ | 0.40 | 0.1542 | $2.5\times10^{-3}$ | **paper setting** |
| $10^{-4}$ | 0.20 | 0.1561 | — | reference |

The paper's setting is $2.5\times10^{-3}$ away from the reference (1.2% in contact area).
In the $\varepsilon$ sweep ($\Delta x=\Delta t=2\times10^{-4}$ fixed),
$\|\eta(T)-\eta_{\mathrm{ref}}\|_\infty$ also decreases monotonically,
$9.7\times10^{-2}\to9.3\times10^{-3}$.

Against the contact-free analytic solution $\eta=h+a\sin(\pi x)e^{\sigma t}\cos(\omega t)$
($\lambda^2+\alpha\pi^2\lambda+\pi^2=0$, $\lambda=-0.049348\pm3.141205i$) the scheme shows
**first-order convergence** (rates 1.20, 1.11, 1.06, 1.04), as predicted when planning.

---

## 5. Issues found and fixed by external review (codex)

### 5.1 First review

A `codex exec` review found the issues below, all fixed. The difference equation and the
energy identity themselves were confirmed correct by independent verification (residual
$5.1\times10^{-15}$ for the difference equation, $4.6\times10^{-11}$ for the identity).

| # | issue | fix |
|---|---|---|
| 1 | contact area overestimated when `store_every` does not divide the number of steps (3× at `store_every=1000`) | `time_weights()` built from `result.t` with the trapezoidal rule |
| 2 | initial data hard-coded endpoint height 1; for `h≠1` this injects $\sim(1-h)^2/\Delta x$ of spurious energy | initial data changed to $h+\text{shape}(x)$; `solve()` warns on mismatch. **The $h$-sensitivity conclusion was corrected** (§2 (a) above) |
| 3 | `contact_dissipation` was reported as "dissipation ≥ 0" although individual steps are not non-negative | renamed to `contact_work`, non-negativity claim withdrawn (the time integral is positive in every run of this repository, but carries no guarantee; §5.2 #3) |
| 4 | 4-connectivity in `ndimage.label` can cut a diagonally advancing contact front | default changed to 8-connectivity (`connectivity=2`), selectable via `--connectivity` |
| 5 | `relative_drift` reached $5\times10^{279}$ for a rest state (denominator exactly 0) | normalized by the energy scale; `absolute_drift` added |
| 6 | `.npz` overwritten under the same name across different parameters; `save()` did not return numpy's actual `.npz` path | parameters encoded in the file name, saved under `--out`; `save()` returns the actual path |
| 7 | `round()` in `scaled_min_size` does not guarantee the area lower bound | changed to `ceil()` |

The roundtrip test was also made an exact all-field comparison via `np.array_equal`, and
regression tests were added for #1, #2 and #5.

### 5.2 Second review (2026-08-29, whole repository)

No high-severity defect (the stencil and the tridiagonal assembly were judged to match §6
of the paper). The 14 findings (8 medium / 6 low) were addressed as follows.

| # | issue | fix |
|---|---|---|
| 1 | the default "contact set" $\{\eta<0\}$ was described as the set where the penalty acts, but the force also requires $v<0$ (about 85% of penetrating nodes carry zero force) | re-described as the penetration set (docstrings, §2 (e)) |
| 2 | "$Q_{\mathrm{num}}$ is $O(\Delta t)$ and vanishes" is overstated (14–18% of the total loss at the paper's resolution); the GIF's dissipation curve showed only viscous + contact | claim qualified (§1.1); the GIF now shows the complete budget viscous + contact + numerical |
| 3 | the time integral of the contact work was asserted positive although no sign is guaranteed | changed to "positive in every run here (an observation, not a theorem)" |
| 4 | $\Delta t/\varepsilon\ge1$ was uniformly labeled UNSTABLE (conflating monotonicity with linear stability) | split into `is_penalty_monotone()` (< 1) and `is_penalty_linearly_stable()` (< 2); labels distinguish NON-MONOTONE / UNSTABLE |
| 5 | the convergence script read as if it separated convergence in Δt, Δx and ε | described explicitly as self-convergence ($\Delta x=\Delta t$ refined jointly + ε sweep at fixed grid) (§4) |
| 6 | the verbatim Ex. 2 profile being negative on $(0.5,0.8)$ (minimum −1, 30% initially penetrating) was undocumented | added to §2 (b), README and docstrings; pinned by tests |
| 7 | the `.npz` name did not include `store_every`, allowing silent overwrites | `_se{store_every}` appended to the file name (with a regression test) |
| 8 | the (A2) test only looked at a spatially summed scalar | replaced by a test that independently reconstructs $P^i$ and $Q_{\mathrm{con}}$ from the snapshots and checks the recorded budget against them |
| 9 | the second assert of the connectivity test was tautological | strict check with a synthetic diagonal band (1 component vs $n$ cells); CLI smoke tests and a stencil-residual test also added |
| 10 | README overstated the contact set as "identical" | changed to visual-match language; the comparison being by eye and Ex. 2 running on inferred data are stated |
| 11 | leftover "Phase" terminology from the deleted plan and a stale test count of 45 | descriptions updated |
| 12 | "in both examples the whole string falls almost uniformly" and "$v^0$ is −50 at the endpoints" do not hold for Ex. 2 | Ex. 1 (uniform fall, 1 component) and Ex. 2 (right 40% falls 100× slower, 2 components) distinguished; endpoints described as "nonzero" |
| 13 | no URLs in the package metadata (broken relative README links in the built package were also noted) | `[project.urls]` added; a license declaration needs a choice and is left open |
| 14 | the threshold-mode tolerance could not be changed from the CLI | `--contact-tol` added and propagated to summary, figures and GIF |

## 6. Example 3 (ours) — a rolling contact front

In the paper's two examples the fast-falling part of the string touches down almost at
once, so the contact set is a "shrinking triangle" in both (Ex. 1: the whole string falls
uniformly, 1 component; Ex. 2: the right 40% falls 100× slower, 2 components). With the
validated solver we built one example where **contact happens in a different pattern**
(`results/ex3/`):

$$
\eta^0(x)=h+\tfrac32\sin(\pi x),\qquad
v^0(x)=-110\,\sin(\pi x)\cdot\tfrac12\Big(1-\tanh\tfrac{x-0.35}{0.1}\Big),\qquad T=0.7,
$$

with the paper's discretization parameters ($\Delta x=\Delta t=1/5000$, $\alpha=0.01$,
$\varepsilon=5\times10^{-4}$, $h=1$).

### 6.1 Design intent

- **Concentrate the downward velocity on the left half**: approximately $-110\sin(\pi x)$
  for $x\le0.35$, essentially at rest ($|v^0|<0.1$) for $x\ge0.7$. The left side touches
  down first and sticks inelastically; the disturbance released by the impact travels right
  and lays the string down progressively.
- **The initial data is fully compatible with (1.3)**: $\eta^0(0)=\eta^0(l)=h$ and also
  $v^0(0)=v^0(l)=0$. The paper's two examples have $v^0$ nonzero at the endpoints
  ($-50$ at both ends in Ex. 1; $-50$ on the left and $-0.5$ on the right in Ex. 2), so an
  initial layer forms there when `solve()` clamps the endpoints to 0. Example 3 has none
  (the $\sin(\pi x)$ factor takes care of it).
- $|v^0|_{\max}=68.8$ (at $x\approx0.26$), still within the monotone-decay condition
  $\Delta t/\varepsilon=0.4$.

### 6.2 Results

| item | value |
|---|---|
| first contact | $t=0.0304$ |
| end of contact | $t=0.5754$ |
| connected components | **1**, $x\in[0.063,0.878]$, a band crossing the $(t,x)$ plane diagonally |
| front speed (linear fit on $t\in[0.1,0.5]$) | left edge 1.46, right edge 1.24 |
| contact width | $t=0.1$: 0.32, $t=0.3$: 0.18, $t=0.5$: 0.22 |
| contact area | 0.1238 |
| penetration depth | $2.79\times10^{-2}=55.8\,\varepsilon$ ($\approx|v^0|_{\max}\varepsilon$, as predicted in §1.3) |
| energy | $E(0)=504.3\to E(T)=7.86$ (viscous 155.7 / contact 273.3 / numerical 67.4), drift $8.0\times10^{-10}$ |
| wall time | 0.66 s |

The front speed slightly exceeding the characteristic speed 1 of the wave part (1.24–1.46)
is due to the viscous term $\alpha\partial_{txx}\eta$ being parabolic, which puts no bound
on the propagation speed of high-frequency components. In the velocity field
(`velocity.png`) it shows up as a straight front separating falling ($\approx-50$) from
in-contact ($\approx0$).

The contact set is a single connected component with smooth boundary, showing that the
observation of Remark 2.2 — "the contact set is quite regular" — also holds for a contact
pattern different from the paper's two examples.

## 7. Animation

`src/contact_damped_wave/animation.py` (`cdw animate`). The GIF is written with
matplotlib's `PillowWriter`, so no external encoder (ffmpeg etc.) is needed. Three panels:

1. The string $\eta(t,\cdot)$ above the obstacle; the part in contact ($\{\eta<0\}$)
   highlighted with a thick black line, the initial shape shown dashed.
2. The contact set in the $(t,x)$ plane, revealed up to the current time. The future is
   covered by a white rectangle (re-uploading the $(n_{\text{stored}},N+1)$ image every
   frame would be heavy, so only the rectangle moves).
3. Energy $E(t)$ and the cumulative dissipation (viscous + contact + numerical: as per
   §1.1 the numerical dissipation is 14–18% of the loss, so the complete budget is shown —
   the two curves are exact mirror images). The energy visibly drops at the moments of
   contact.

Outputs: `results/ex1/animation.gif` (1.5 MB), `results/ex2/animation.gif` (1.4 MB),
`results/ex3/animation.gif` (2.1 MB, 180 frames, 765×544). The frames are subsampled from
the stored snapshots of a `Result`, so a run made with a large `--store-every` animates
just as well.

## 8. Remaining work

- Numerical check that the contact dissipation concentrates at the contact boundary
  (Thm 2.3).
- Automated side-by-side comparison images against the paper's figures (currently compared
  by eye).
- Convergence tests separating the time, space and penalization errors (independent
  $\Delta x$/$\Delta t$ sweeps and an $\varepsilon$ refinement with
  $\Delta t/\varepsilon\to0$; see the note in §4).
- A license declaration for the package (awaiting a choice).
