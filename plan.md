# plan.md — arXiv:2412.06185 数値実験の再現計画

## 0. 概要

- **対象論文**: B. Muha, S. Trifunović,
  *Analysis of an Inelastic Contact Problem for the Damped Wave Equation*,
  arXiv:2412.06185v2 (2025-05-15). <https://arxiv.org/abs/2412.06185>
- **ゴール**: 論文 Section 6 "Numerical examples" の **Example 1 / Example 2**（Figures 2–5）を
  Python (numpy / scipy / matplotlib) + uv で再現する。
- **前提**: 元実装は MATLAB で非公開。本文に記載された差分スキームとパラメータのみから再構成する。
- 論文 PDF は `paper/2412.06185v2.pdf`、図のページ画像は `paper/pages/page21-24.png`（いずれも git 管理外。
  README の手順で再取得可能）。

---

## 1. 対象問題（論文 §1.1）

長さ $l$ の粘弾性弦の鉛直変位 $\eta:(0,\infty)\times(0,l)\to\mathbb{R}$。障害物平面 $y=0$ の上で振動し、
接触は完全非弾性（接触後に速度が消える）。

$$
\begin{aligned}
&\eta \ge 0 && \text{(非貫入条件, 1.1)}\\
&\partial_{tt}\eta - \partial_{txx}\eta - \partial_{xx}\eta = F_{\mathrm{con}} && \text{(減衰波動方程式, 1.2)}\\
&\eta(t,0)=\eta(t,l)=h>0 && \text{(両端固定, 1.3)}\\
&\frac{d}{dt}\Big(\tfrac12\!\int_0^l|\partial_t\eta|^2 + \tfrac12\!\int_0^l|\partial_x\eta|^2\Big)
 + \int_0^l|\partial_{tx}\eta|^2 + \int_0^l D_{\mathrm{con}} = 0 && \text{(エネルギー収支, 1.4)}
\end{aligned}
$$

- $F_{\mathrm{con}}\ge0$: 接触反力（特異測度）, $D_{\mathrm{con}}\ge0$: 接触による散逸。
- (A2) $\operatorname{supp}D_{\mathrm{con}}\subseteq\operatorname{supp}F_{\mathrm{con}}\subseteq\partial\{\eta=0\}$（Signorini 条件）
- (A3) 弦が下向きに動いていないところでは反力なし、(A4) 接触後の速度は 0（跳ね返らない）。
- 存在証明は **ペナルティ法**: 反発力 $\frac1\varepsilon\chi_{\{\eta<0\}}(\partial_t\eta)^-$ を加えた近似問題
  （§3.1, (3.1)）の $\varepsilon\to0$ 極限。ここで $f^-=\max\{0,-f\}$。
- Remark 2.2: 「数値データはいずれも接触集合がかなり正則であることを示唆する」→ その根拠が §6 の 2 例。

---

## 2. 論文の数値スキーム（§6, 転記）

論文の離散化（$\alpha>0$ は粘弾性係数、$\Delta t,\Delta x$ は刻み幅）:

$$
\frac{\eta^{i+1}_j-2\eta^i_j+\eta^{i-1}_j}{(\Delta t)^2}
-\frac{\alpha}{\Delta t}\left(
 \frac{\eta^{i+1}_{j+1}-2\eta^{i+1}_j+\eta^{i+1}_{j-1}}{(\Delta x)^2}
-\frac{\eta^{i}_{j+1}-2\eta^{i}_j+\eta^{i}_{j-1}}{(\Delta x)^2}\right)
-\frac{\eta^{i+1}_{j+1}-2\eta^{i+1}_j+\eta^{i+1}_{j-1}}{(\Delta x)^2}
=\frac1\varepsilon\,\chi_{\{\eta^i_j<0\}}\left(\frac{\eta^i_j-\eta^{i-1}_j}{\Delta t}\right)^{-}
$$

$$
\eta^i_0=\eta^i_N=0\ (\text{原文ママ。実際は } h,\ \S4 \text{ 参照}),\qquad
N=l/\Delta x,\quad M=T/\Delta t,\quad \eta^i=[\eta^i_0,\dots,\eta^i_N]^T .
$$

- 空間: 2 階中心差分。時間: $\partial_{tt}$ は中心差分、$\partial_{xx}$ は **陰的**（$i+1$ レベル）、
  $\partial_{txx}$ は $(\delta_x^2\eta^{i+1}-\delta_x^2\eta^i)/\Delta t$（陰的）、
  **ペナルティ項は前ステップの値から陽的**に評価（"taken explicitly from the previous time step"）。
- 論文は MATLAB で計算し、$\Delta t,\Delta x,\varepsilon$ に関する数値的収束を確認したと述べている
  （具体的な収束データは掲載なし）。

### 2.1 実装用の整理

$D$ を Dirichlet 境界付き離散ラプラシアン（$(D\eta)_j=(\eta_{j+1}-2\eta_j+\eta_{j-1})/\Delta x^2$）、
ペナルティを $P^i_j=\frac1\varepsilon\,\mathbf 1_{\{\eta^i_j<0\}}\max\{0,-(\eta^i_j-\eta^{i-1}_j)/\Delta t\}$
とすると、各ステップは **定数係数の三重対角線形系**:

$$
\big[I-(\alpha\Delta t+\Delta t^2)D\big]\eta^{i+1}
= 2\eta^i-\eta^{i-1}-\alpha\Delta t\,D\eta^i+\Delta t^2P^i .
$$

論文のパラメータ（$\Delta t=\Delta x=1/5000,\ \alpha=0.01$）では
$\alpha\Delta t/\Delta x^2=50$, $\Delta t^2/\Delta x^2=1$ なので係数行列は
$\mathrm{tridiag}(-51,\ 103,\ -51)$（対角優位・SPD）。1 ステップ $O(N)$。

- **陽的ペナルティの安定性**: 貫入中 ($\eta<0,\ v<0$) の速度更新は近似的に
  $v^{i+1}\approx(1-\Delta t/\varepsilon)\,v^i$。論文設定では $\Delta t/\varepsilon=0.4$ なので
  単調に $0$ に減衰（跳ね返りなし）。$\Delta t/\varepsilon\ge1$ で振動、$\ge2$ で不安定になるため、
  $\varepsilon$ の収束テストでは **$\Delta t/\varepsilon<1$ を保つ**こと。
- 時間精度: $\partial_{xx}$ の陰的評価と $\partial_{txx}$ の片側評価により **全体 1 次精度**（$\Delta t$）。

---

## 3. 数値例の設定（§6.1, §6.2）

共通: $l=1$, $\Delta t=\Delta x=1/5000$（$N=5000$）, $\alpha=0.01$, $\varepsilon=0.0005$, $h=1$（図より）。

### Example 1（Fig. 2, 3）
- $T=0.3$（$M=1500$）
- $\eta^0(x)=1+\tfrac12\sin^2(10\pi x)$, $\ v^0\equiv-50$
- **Fig. 2**: $t=0,\ 0.02,\ 0.04,\ 0.06,\ 0.2,\ 0.3$ の $\eta$ のスナップショット（2×3 パネル、
  縦軸 $[-0.5,2]$、障害物 $y=0$ を水平線で表示）。
- **Fig. 3 左**: $(t,x)$ 平面上の接触集合（黒塗り）。$t\approx0.02$ で $x\in[\sim0.05,\sim0.95]$ に
  一気に形成（左縁は $\sin^2$ 由来の 10 個の波打ち）、その後両側からほぼ直線的に縮み
  $t\approx0.26$ に $x=0.5$ 付近で消滅。$x=0.5$ に関して対称。
- **Fig. 3 右**: $(t,x)$ 平面上の速度場 $\partial_t\eta$ のカラーマップ（カラーバー $-60\sim10$）。
  接触前は $\approx-50$、接触領域内は $\approx0$、離脱フロントに沿って正の帯（$\sim10$）。

### Example 2（Fig. 4, 5）
- $T=0.5$（$M=2500$）
- 本文の式（**原文ママ**）:
  $\eta^0=x\ (0\le x<0.2),\ \sin(\pi(x-0.2)/0.3)\ (0.2\le x<0.8),\ 2-x\ (0.8\le x<1)$
- 図 4(a) から読み取れる実際の初期データ（**本プロジェクトの採用値**、§4 参照）:
  $$
  \eta^0(x)=\begin{cases}
  1+5x, & 0\le x<0.2,\\
  2+\sin\!\big(\pi(x-0.2)/0.3\big), & 0.2\le x<0.8,\\
  6-5x, & 0.8\le x\le1,
  \end{cases}
  \qquad
  v^0(x)=\begin{cases}-50, & 0\le x<0.6,\\ -0.5, & 0.6\le x\le1.\end{cases}
  $$
  （$\eta^0(0)=\eta^0(1)=1$, $\eta^0(0.2)=\eta^0(0.5)=\eta^0(0.8)=2$, 最大 3 at $x=0.35$, 極小 1 at $x=0.65$。
  連続で境界値 $h=1$ と整合。）
- **Fig. 4**: $t=0,\ 0.04,\ 0.08,\ 0.16,\ 0.28,\ 0.32$ のスナップショット（縦軸 $[-0.5,3]$）。
  キャプションは "example 1" とあるが example 2 の誤植。
- **Fig. 5 左**: 接触集合は **2 つの連結成分**。大きい成分: $x\in[\sim0.1,\sim0.6]$, $t\in[\sim0.02,\sim0.26]$
  （右向きの三角形状、先端 $x\approx0.35$–$0.4$）。小さい成分: $x\in[\sim0.65,\sim0.85]$, $t\in[\sim0.25,\sim0.37]$。
- **Fig. 5 右**: 速度場カラーマップ（カラーバー $-50\sim40$）。

---

## 4. 論文記述の不整合と本プロジェクトの採用方針

| # | 論文の記述 | 問題 | 採用方針 |
|---|-----------|------|---------|
| (a) | $\eta^i_0=\eta^i_N=0$ | PDE の境界条件は $\eta=h$。Fig. 2/4 では常に端点値 1 | **$\eta^i_0=\eta^i_N=h=1$** を採用（$h$ はパラメータ化） |
| (b) | Ex.2 の $\eta^0$: `x`, `sin(...)`, `2−x` | 不連続（0.2 で 0.2→0、0.8 で 0→1.2）、$\eta^0(0)=0$ で仮定 $\eta_0\ge c>0$ にも反する。Fig. 4(a) と不一致 | **図に整合する式（§3）を既定**とし、原文式も `--initial paper-literal` で選択可能にして差を記録 |
| (c) | 添字 "$0\le i\le l/\Delta x$, $1\le j\le T/\Delta t$" | 式中では上付き $i$ が時間、下付き $j$ が空間 | **上付き = 時間、下付き = 空間** |
| (d) | 初期ステップ（$\eta^{-1}$ または $\eta^1$）の与え方が未記載 | 中心差分 + 陽的ペナルティのため $\eta^{-1}$ が必要 | **$\eta^{-1}:=\eta^0-\Delta t\,v^0$** を既定（これで step 0 のペナルティ速度が $v^0$ になる）。代替 $\eta^1=\eta^0+\Delta t v^0$ との差を感度テストで確認 |
| (e) | 接触集合の描画基準が未記載 | $\{\eta<0\}$ か $\{\eta\le\text{tol}\}$ か | 既定 $\{\eta^i_j<0\}$（ペナルティの発動集合と一致）。$\{\eta\le\text{tol}\}$ 版も出力し比較 |
| (f) | Fig. 4 キャプション "example 1" | 誤植 | 無視（example 2 として扱う） |

---

## 5. 実装計画

### Phase 0 — 環境構築 ✅（本コミット）
- `uv init --lib --python 3.12`、依存: numpy / scipy / matplotlib、dev: pytest / ruff。
- `plan.md`, `README.md`, `.gitignore`, `paper/`（PDF, ページ画像）。

### Phase 1 — コアソルバー `src/contact_damped_wave/`
| モジュール | 内容 |
|-----------|------|
| `params.py` | `@dataclass(frozen=True) Params(l, T, dx, dt, alpha, eps, h)`; `N`, `M` プロパティ; 論文設定 `EXAMPLE1`, `EXAMPLE2` の定数 |
| `initial_data.py` | `example1_eta0/v0`, `example2_eta0/v0`（figure-consistent, paper-literal の両方）; 格子 `x = linspace(0, l, N+1)` |
| `solver.py` | 三重対角行列の構築（`scipy.linalg.solve_banded` 用の `ab` 配列、または `scipy.sparse.linalg.splu` で一度だけ分解）、`step(eta_prev, eta_cur) -> eta_next`、`solve(params, eta0, v0, *, store_every=1) -> Result` |
| `diagnostics.py` | 離散エネルギー $E^i=\tfrac12\lVert v^i\rVert^2_{\Delta x}+\tfrac12\lVert \delta_x\eta^i\rVert^2_{\Delta x}$、粘性散逸 $\Delta x\Delta t\sum\lvert\delta_x v\rvert^2$、ペナルティ散逸 $\Delta x\Delta t\sum P^i v^i$、接触集合マスク、連結成分数、接触開始/終了時刻 |
| `plotting.py` | `plot_snapshots(result, times, ylim)`（Fig. 2/4 相当）、`plot_contact_set(result)`、`plot_velocity_field(result, vlim)`（Fig. 3/5 相当） |
| `cli.py` | `uv run cdw run --example 1 --out results/ex1`、`--dt --dx --eps --initial` の上書き、結果を `.npz` 保存 |

- 履歴配列: Ex.2 で $(2501\times5001)$ float64 ≈ 100 MB → 既定は全保存、`store_every` で間引き可。
- 速度場は履歴から差分で計算（保存しない）。
- 計算量: 1 ステップ $O(N)$ の帯行列解法 → 全体で数秒以内（見積り）。

### Phase 2 — 検証（`tests/`）
1. **障害物なしの解析解比較**: $\eta=h+a\sin(k\pi x/l)e^{\lambda t}$, $\lambda^2+\alpha(k\pi/l)^2\lambda+(k\pi/l)^2=0$
   （$\alpha=0.01,k=1$: $\lambda\approx-0.049\pm3.141i$）。$h=1,a=0.1$ で接触なし。
   $\Delta t=\Delta x$ を細分して **1 次収束**を確認。
2. **エネルギー不等式**: ペナルティ・粘性散逸を含めた離散エネルギー収支が（丸めを除き）非増加。
3. **対称性**: Example 1 で $\eta^i_j=\eta^i_{N-j}$（機械精度）。
4. **非貫入の近似**: $\min\eta\ge-C\varepsilon$ 程度（$\varepsilon$ を減らすと貫入深さも減る）。
5. **静止解**: $v^0=0$, $\eta^0\equiv h$ が不変。
6. **収束テスト（論文が実施したと述べているもの）**: $\Delta t=\Delta x\in\{1/1000,1/2000,1/5000\}$,
   $\varepsilon\in\{0.002,0.001,0.0005\}$（$\Delta t/\varepsilon<1$ を維持）で接触集合の面積・接触開始時刻・
   $t=T$ の $\eta$ を比較し、表にまとめる。

### Phase 3 — Example 1 の再現
- `uv run cdw run --example 1` → `results/ex1/fig2_snapshots.png`, `fig3_contact_set.png`, `fig3_velocity.png`。
- チェックリスト（§3 の Fig. 2/3 の記述と突き合わせ）:
  - [ ] $t=0.02$ で $\eta$ の極小が $0$ に到達（$v^0=-50$ より $1/50=0.02$）
  - [ ] $t=0.04$–$0.06$ で $x\in[\sim0.05,\sim0.95]$ がほぼ接触、端点付近だけ 1 に立ち上がる
  - [ ] $t=0.2$ で接触集合 $\approx[0.3,0.7]$、$t=0.3$ で $\approx[0.45,0.55]$ 以下（Fig. 3 では 0.26 で消滅）
  - [ ] 接触集合が $x=0.5$ に関して対称、左縁が波打つ
  - [ ] 速度場の値域が $[-60,10]$ 程度、離脱フロントに正の帯

### Phase 4 — Example 2 の再現
- `uv run cdw run --example 2` → `results/ex2/fig4_snapshots.png`, `fig5_contact_set.png`, `fig5_velocity.png`。
- チェックリスト:
  - [ ] $t=0.04$ で $x\approx[0.03,0.18]$ と $[0.5,0.58]$ の 2 か所が接触、右側の山（$x=0.8$）は $\approx1.75$
  - [ ] $t=0.08$ で $[\sim0.1,\sim0.6]$ が接触、$t=0.16$ で $[\sim0.2,\sim0.5]$
  - [ ] $t=0.28$–$0.32$ で $x\approx0.75$–$0.85$ に新たな接触（小さい成分）
  - [ ] 接触集合の連結成分が 2 個（大: $[0.1,0.6]\times[0.02,0.26]$、小: $[0.65,0.85]\times[0.25,0.37]$）
  - [ ] 速度場の値域が $[-50,40]$ 程度
- 原文式（paper-literal）の初期データでも実行し、図と一致しないことを記録（不整合 (b) の裏付け）。

### Phase 5 — 追加解析（任意）
- エネルギーの時間推移（運動 / 弾性 / 粘性散逸累積 / 接触散逸累積）を図示し、(1.4) と
  「散逸は接触時のみ・接触境界に集中」（Theorem 2.3）を数値的に確認。
- $\varepsilon\to0$ での貫入深さ、接触散逸の収束。
- 初期ステップの与え方 (d) と接触集合の閾値 (e) の感度。

### Phase 6 — レポート
- `results/figures/` に最終図、`README.md` に論文図との並列比較表と再現度の評価を記載。
- 不整合 (a)–(f) と採用方針、収束テスト結果を `docs/notes.md` にまとめる。

---

## 6. ディレクトリ構成（計画）

```
.
├── plan.md                  # 本ファイル
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── paper/                   # 論文 PDF・ページ画像（git 管理外）
├── src/contact_damped_wave/
│   ├── __init__.py
│   ├── params.py
│   ├── initial_data.py
│   ├── solver.py
│   ├── diagnostics.py
│   ├── plotting.py
│   └── cli.py
├── tests/
│   ├── test_solver_exact.py     # 解析解との比較・収束
│   ├── test_energy.py
│   ├── test_symmetry.py
│   └── test_initial_data.py
├── scripts/
│   ├── run_examples.py          # Ex.1/2 を一括実行
│   └── convergence_study.py
├── results/
│   ├── data/                    # .npz（git 管理外）
│   └── figures/
└── docs/notes.md
```

---

## 7. 再現成功の判定基準

論文には数値表がなく図のみのため、判定は**定性的一致 + 自前の検証**で行う。

1. Phase 2 のテストがすべて通る（解析解 1 次収束、エネルギー非増加、対称性）。
2. Fig. 2–5 の各パネルと生成図を並べたとき、接触開始時刻・接触集合の形状/連結成分数/消滅時刻・
   速度場の値域が §3 の記述の範囲で一致する。
3. $\Delta t,\Delta x,\varepsilon$ を変えても接触集合の形状が安定（収束表）。

---

## 8. リスク・注意点

- 論文式 (b) の解釈が間違っていた場合、Ex.2 の形状が合わない → paper-literal 版も必ず実行して比較。
- 初期ステップ (d) の違いは 1 ステップ分（$O(\Delta t)$）のずれに留まる見込みだが、接触開始時刻に影響し得る。
- 接触集合の見た目は閾値 (e) に敏感な可能性 → 2 通り出力。
- MATLAB との丸め・線形ソルバの差は無視できる想定（SPD 三重対角）。
- 履歴の全保存で Ex.2 は ~100 MB → メモリが厳しければ `store_every` で間引く。
