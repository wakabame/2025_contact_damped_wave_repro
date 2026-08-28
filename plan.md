# plan.md — arXiv:2412.06185 数値実験の再現計画と実施記録

> **状態（最終更新 2026-08-28）**
>
> | Phase | 内容 | 状態 |
> |---|---|---|
> | 0 | 環境構築 | ✅ 完了 |
> | 1 | コアソルバー実装 | ✅ 完了 |
> | 2 | 検証テスト | ✅ 完了（52 件、3.3 秒）|
> | 3 | Example 1 の再現（Fig. 2, 3）| ✅ 完了（差異 1 点、§7.1）|
> | 4 | Example 2 の再現（Fig. 4, 5）| ✅ 完了（初期データの読み替えが必要、§4 (b)）|
> | 5 | 追加解析（感度・収束）| 🟡 大部分実施（残: 接触散逸の局在の確認）|
> | 6 | レポート | 🟡 README / docs/notes.md 済み（残: 論文図との自動並列比較）|
> | 7 | **独自例（Example 3）と GIF アニメーション** | ✅ 完了（§9）|
>
> 実測値の詳細は [`docs/notes.md`](docs/notes.md)、生成物は `results/`。
> 本ファイルは計画と、その計画に対して**実際に何が起きたか**を併記する。

## 0. 概要

- **対象論文**: B. Muha, S. Trifunović,
  *Analysis of an Inelastic Contact Problem for the Damped Wave Equation*,
  arXiv:2412.06185v2 (2025-05-15). <https://arxiv.org/abs/2412.06185>
- **ゴール**: 論文 Section 6 "Numerical examples" の **Example 1 / Example 2**（Figures 2–5）を
  Python (numpy / scipy / matplotlib) + uv で再現する。
  → 達成。さらに検証済みソルバーを使った独自例（Example 3）とアニメーションを追加（Phase 7）。
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
  （具体的な収束データは掲載なし）→ 本プロジェクトで実測（§7.3, `results/convergence/`）。

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

- ✅ **実装**: `cholesky_banded` で 1 度だけ分解し、各ステップ `cho_solve_banded`。
  論文設定（$N=5000$）で Example 1 が 0.30 秒、Example 2 が 0.48 秒、Example 3 が 0.66 秒。
  独立検証で差分式の残差 $5.1\times10^{-15}$。
- ✅ **陽的ペナルティの安定性**（計画時の予測どおり）: 貫入中 ($\eta<0,\ v<0$) の速度更新は近似的に
  $v^{i+1}\approx(1-\Delta t/\varepsilon)\,v^i$。論文設定では $\Delta t/\varepsilon=0.4$ なので
  単調に $0$ に減衰（跳ね返りなし）。**$\Delta t/\varepsilon\ge1$ で振動、$\ge2$ で発散を実測**
  （docs/notes.md §1.2 の表: $\Delta t/\varepsilon=2$ で $E(T)$ が 24.6 → 432.3、接触散逸が負）。
  この条件は**論文に記載がなく**、$\varepsilon\to0$ と $\Delta t\to0$ を独立に取れないことを意味する。
  `Params.is_penalty_stable()` として実装し、CLI が警告を出す。
- ✅ **時間精度**: 障害物に触れない解析解との比較で **1 次収束**（rate 1.20, 1.11, 1.06, 1.04）。予測どおり。

### 2.2 離散エネルギー恒等式（実装時に導出。計画にはなかった）

$v^{i+1/2}=(\eta^{i+1}-\eta^i)/\Delta t$ を掛けて和をとると（端点で $v=0$ なので部分和分は厳密）、
$E^i=\frac{\Delta x}2\sum_j(v^{i-1/2}_j)^2+\frac{\Delta x}2\sum_j(\delta_x\eta^i_j)^2$ に対し **厳密な**恒等式

$$
E^{i+1}-E^i=-\Delta t\,[\,Q_{\mathrm{visc}}+Q_{\mathrm{num}}+Q_{\mathrm{con}}\,]
$$

が成り立つ（$Q_{\mathrm{visc}},Q_{\mathrm{num}}\ge0$ は平方和、$Q_{\mathrm{con}}=-\Delta x\sum_jP^i_jv^{i+1/2}_j$）。
$Q_{\mathrm{num}}$ はスキーム由来の $O(\Delta t)$ の数値散逸で、(1.4) には対応物がない。
ソルバーが 3 項を毎ステップ記録するので収支は**機械精度で閉じる**（drift $10^{-10}$ 台）。
式の詳細と、当初これを台形則で近似して drift 17.8 を出した失敗は docs/notes.md §1.1。

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

## 4. 論文記述の不整合と本プロジェクトの採用方針（実施結果つき）

| # | 論文の記述 | 問題 | 採用方針 | 実施結果 |
|---|-----------|------|---------|---------|
| (a) | $\eta^i_0=\eta^i_N=0$ | PDE の境界条件は $\eta=h$。Fig. 2/4 では常に端点値 1 | **$\eta^i_0=\eta^i_N=h=1$** を採用（$h$ はパラメータ化） | ✅ 採用。初期データを $h+\text{shape}(x)$ の形にして任意の $h$ で (1.3) と整合させた。$h$ 感度は docs/notes.md §2 (a)（$E(0)$ は $h$ 非依存、初回接触時刻は単調、消滅時刻は非単調）|
| (b) | Ex.2 の $\eta^0$: `x`, `sin(...)`, `2−x` | 不連続（0.2 で 0.2→0、0.8 で 0→1.2）、$\eta^0(0)=0$ で仮定 $\eta_0\ge c>0$ にも反する。Fig. 4(a) と不一致 | **図に整合する式（§3）を既定**とし、原文式も `--initial paper-literal` で選択可能にして差を記録 | ✅ **決定的**。原文式では $E(0)$ が 9 倍（771 → 6978）、$t=0$ から接触、貫入 1.0（$2000\varepsilon$）で Fig. 4 を再現できない。`results/ex2_paper_literal/` に証拠を保存 |
| (c) | 添字 "$0\le i\le l/\Delta x$, $1\le j\le T/\Delta t$" | 式中では上付き $i$ が時間、下付き $j$ が空間 | **上付き = 時間、下付き = 空間** | ✅ この解釈で整合 |
| (d) | 初期ステップ（$\eta^{-1}$ または $\eta^1$）の与え方が未記載 | 中心差分 + 陽的ペナルティのため $\eta^{-1}$ が必要 | **$\eta^{-1}:=\eta^0-\Delta t\,v^0$** を既定。代替 $\eta^1=\eta^0+\Delta t v^0$ との差を感度テストで確認 | ✅ 両方実装（`--initial-step`）。どちらもエネルギー収支は機械精度で閉じ、結果の差は $O(\Delta t)$ |
| (e) | 接触集合の描画基準が未記載 | $\{\eta<0\}$ か $\{\eta\le\text{tol}\}$ か | 既定 $\{\eta^i_j<0\}$。$\{\eta\le\text{tol}\}$ 版も出力し比較 | ✅ 差は**面積比 1% 未満**。計画が懸念したほど敏感ではなかった |
| (f) | Fig. 4 キャプション "example 1" | 誤植 | 無視（example 2 として扱う） | ✅ そのまま |
| **(g)** | ペナルティ項の安定条件 | **記載なし**（新たに発見） | — | ⚠️ $\Delta t/\varepsilon<1$ が必要。$\ge2$ でエネルギーが増加し破綻。論文設定 0.4 は安全側だが、$\varepsilon$ を細かくするには $\Delta t$ も細かくする必要がある（§2.1, docs/notes.md §1.2）|

---

## 5. 実装計画と実施状況

### Phase 0 — 環境構築 ✅
- `uv init --lib --python 3.12`、依存: numpy / scipy / matplotlib / pillow、dev: pytest / ruff。
- `plan.md`, `README.md`, `.gitignore`, `paper/`（PDF, ページ画像）。

### Phase 1 — コアソルバー `src/contact_damped_wave/` ✅
| モジュール | 内容 | 実施 |
|-----------|------|------|
| `params.py` | `@dataclass(frozen=True) Params(length, T, dx, dt, alpha, eps, h)`; `N`, `M`; 論文設定 `EXAMPLE1`, `EXAMPLE2` | ✅ + `penalty_ratio` / `is_penalty_stable()` / `implicit_coefficient`、独自例の `EXAMPLE3` |
| `initial_data.py` | `example1_eta0/v0`, `example2_eta0/v0`（figure, paper-literal）; 格子 | ✅ + `example3_eta0/v0`（Phase 7）。全て $h+\text{shape}$ 形式 |
| `solver.py` | 帯行列の Cholesky 分解 + 各ステップ $O(N)$、`solve(...) -> Result` | ✅ + §2.2 の厳密なエネルギー収支を毎ステップ記録、`Result.save/load` |
| `diagnostics.py` | 離散エネルギー、粘性/ペナルティ散逸、接触集合、連結成分、接触開始/終了時刻 | ✅ + `time_weights()`（`store_every` が割り切らない場合の台形則）、`connectivity` 選択 |
| `plotting.py` | Fig. 2/4 相当、接触集合、速度場 | ✅ + `plot_energy` |
| `animation.py` | — | ✅ **計画外・Phase 7 で追加**: GIF アニメーション（PillowWriter、外部エンコーダ不要）|
| `cli.py` | `cdw run --example N --out ...` | ✅ + `cdw animate`、`--example 3` |

- 履歴配列: 既定は全保存、`store_every` で間引き可（Ex.3 は `store_every=5` で 28 MB/場）。
- 速度場は履歴から差分で計算（保存はしないが `Result.v` として同時に返す）。
- 計算量: 予測どおり 1 ステップ $O(N)$、全体 1 秒未満。

### Phase 2 — 検証（`tests/`）✅ 52 件 / 3.3 秒
1. ✅ **障害物なしの解析解比較**: $\eta=h+a\sin(\pi x)e^{\sigma t}\cos(\omega t)$
   （$\lambda=-0.049348\pm3.141205i$）で **1 次収束**を確認。
2. ✅ **エネルギー恒等式**: §2.2 の収支が機械精度（相対 drift $<10^{-9}$）で閉じる。
3. ✅ **対称性**: Example 1 で $\eta^i_j=\eta^i_{N-j}$（$1.7\times10^{-12}$）。
4. ✅ **非貫入の近似**: 貫入深さ $\approx|v^0|\varepsilon$（Ex.1 で $2.56\times10^{-2}$ vs 予測 $2.5\times10^{-2}$）。
5. ✅ **静止解**: $v^0=0,\ \eta^0\equiv h$ が不変。
6. ✅ **収束テスト**: `scripts/convergence_study.py`（§7.3）。
7. ✅ 計画外: `store_every` / `h` 不整合 / drift 正規化の回帰テスト（外部レビュー由来、docs/notes.md §5）、
   Example 3 の接触前線が単調に右進すること、GIF 出力のスモークテスト。

### Phase 3 — Example 1 の再現 ✅
`uv run cdw run --example 1` → `results/ex1/`。チェックリストの結果:
  - [x] $t=0.02$ で $\eta$ の極小が $0$ に到達 → 初回接触 **0.0236**（$v^0=-50$ からの素朴な予測 0.02 に近い）
  - [x] $t=0.04$–$0.06$ で $x\in[\sim0.05,\sim0.95]$ がほぼ接触 → 実測 $[0.032,0.968]$
  - [x] $t=0.2$ の接触区間 $\approx[0.3,0.7]$ → 実測 $\approx[0.28,0.75]$
  - [x] 接触集合が $x=0.5$ に関して対称（$1.7\times10^{-12}$）、左縁の波打ち **10 個**
  - [x] 速度場の値域が $[-60,10]$ 程度、離脱フロントに正の帯
  - [ ] **$t\approx0.26$ で消滅 → 実測 0.299**（唯一の差異。§7.1 で分析）

### Phase 4 — Example 2 の再現 ✅
`uv run cdw run --example 2` → `results/ex2/`。チェックリストの結果:
  - [x] $t=0.04$ で 2 か所接触、右側の山が $\approx1.75$
  - [x] $t=0.08$ で $[\sim0.1,\sim0.6]$、$t=0.16$ で $[\sim0.2,\sim0.5]$
  - [x] $t=0.28$–$0.32$ に小さい成分
  - [x] **連結成分 2 個**: 大 $t\in[0.026,0.261]\times x\in[0.032,0.594]$、
        小 $t\in[0.243,0.381]\times x\in[0.693,0.860]$（論文の読み取りとほぼ一致）
  - [x] 速度場の値域が $[-50,40]$ 程度
  - [x] 原文式（paper-literal）でも実行し、図と一致しないことを記録 → §4 (b)

### Phase 5 — 追加解析 🟡
- [x] エネルギーの時間推移（運動+弾性 / 粘性 / 接触 / 数値散逸の累積）を `energy.png` に図示。
- [x] $\varepsilon\to0$ での貫入深さの減少（$4\times10^{-3}\to2.5\times10^{-4}$ で $2.2\times10^{-1}\to1.2\times10^{-2}$）。
- [x] 初期ステップ (d)・接触集合の閾値 (e)・端点高さ $h$・粘性 $\alpha$ の感度（docs/notes.md §2, §3）。
- [ ] **残**: 接触散逸が接触境界 $\partial\{\eta=0\}$ に集中すること（Theorem 2.3 / 仮定 (A2)）の数値的確認。

### Phase 6 — レポート 🟡
- [x] `results/ex{1,2,3}/` に最終図、README に論文図との比較表と再現度の評価。
- [x] 不整合 (a)–(g) と採用方針、収束テスト結果を `docs/notes.md` に整理。
- [ ] **残**: 論文図と生成図を並べた比較画像の自動生成（現状は目視比較）。

### Phase 7 — 独自例とアニメーション ✅（計画外の追加）
検証の済んだソルバーを使い、論文の 2 例とは異なる接触の起こり方を示す例を作って GIF にする。→ §9。

---

## 6. ディレクトリ構成（実際）

```
.
├── plan.md                  # 本ファイル（計画 + 実施記録）
├── README.md
├── docs/notes.md            # 再現ノート（スキーム整理・不整合の裏付け・実測値）
├── pyproject.toml / uv.lock / .python-version
├── paper/                   # 論文 PDF・ページ画像（git 管理外）
├── src/contact_damped_wave/
│   ├── __init__.py / py.typed
│   ├── params.py            # Params, EXAMPLE1/2/3
│   ├── initial_data.py      # 3 例の初期データ（Ex.2 は figure / paper-literal）
│   ├── solver.py            # 三重対角陰解法 + 厳密なエネルギー収支
│   ├── diagnostics.py       # 接触集合・連結成分・貫入・エネルギー収支
│   ├── plotting.py          # Fig. 2–5 相当の静止画
│   ├── animation.py         # GIF アニメーション（Phase 7）
│   └── cli.py               # cdw run / cdw animate
├── tests/                   # 52 件
│   ├── conftest.py          # 粗い格子のフィクスチャ（いずれも dt/eps < 1）
│   ├── test_params.py
│   ├── test_initial_data.py
│   ├── test_solver.py       # 解析解との 1 次収束、roundtrip
│   ├── test_contact.py      # エネルギー恒等式・対称性・貫入・接触集合
│   └── test_animation.py    # GIF 出力
├── scripts/
│   ├── run_examples.py      # 3 例 + 原文式版 + GIF を一括実行
│   └── convergence_study.py # Δt=Δx, ε の収束表
└── results/
    ├── ex1/ ex2/ ex2_paper_literal/ ex3/   # 図・summary.txt・GIF（.npz は git 管理外）
    └── convergence/
```

計画時のテストファイル名（`test_solver_exact.py` / `test_energy.py` / `test_symmetry.py`）は、
検証対象ごとではなく**モジュールごと**に整理し直した（`test_solver.py` / `test_contact.py`）。

---

## 7. 再現成功の判定基準と判定結果

論文には数値表がなく図のみのため、判定は**定性的一致 + 自前の検証**で行う。

1. ✅ Phase 2 のテストがすべて通る（解析解 1 次収束、エネルギー機械精度、対称性 $10^{-12}$）。
2. 🟡 Fig. 2–5 と生成図の一致 — スナップショット 12 枚は全て目視一致、接触集合の形状・連結成分数・
   速度場の値域も一致。**Example 1 の接触消滅時刻のみ差異**（下記 7.1）。
3. ✅ $\Delta t,\Delta x,\varepsilon$ を変えても接触集合の形状は安定（下記 7.3）。

### 7.1 唯一の差異: Example 1 の接触消滅時刻（論文 ≈0.26 / 本再現 0.299）

- 格子・$\varepsilon$ に関して**収束済み**（$\Delta x=10^{-4}$ でも 0.300）。数値誤差ではない。
- $h$・$\alpha$ のどちらにも**非単調**に依存するが、両者とも論文が明示している値なので
  パラメータのずれでは説明できない（docs/notes.md §3）。
- 論文自身の Fig. 2(f)（$t=0.3$）は $x\in[0.42,0.58]$ で $\eta$ が 0 に貼り付いて見えており、
  Fig. 3 の「$t\approx0.26$ で消滅」と**論文内部でも整合しない**。離脱直前の極めて浅い領域
  （本再現では $t=0.27$ で $\min\eta=-2.3\times10^{-3}$）の描画閾値の差と考えられる。
- **パラメータの合わせ込みはしていない。**

### 7.2 定量的な再現結果

| | 論文 | 本再現 |
|---|---|---|
| Ex.1 初回接触 | ≈0.02 | 0.0236 |
| Ex.1 接触集合 | $x=0.5$ 対称の三角形、左縁に 10 個の波打ち | 同一（対称性 $1.7\times10^{-12}$）|
| Ex.1 接触消滅 | ≈0.26 | 0.299（§7.1）|
| Ex.2 連結成分数 | 2 | 2 |
| Ex.2 大成分 | $t\in[\sim0.02,\sim0.26]$, $x\in[\sim0.1,\sim0.6]$ | $t\in[0.026,0.261]$, $x\in[0.032,0.594]$ |
| Ex.2 小成分 | $t\in[\sim0.25,\sim0.37]$, $x\in[\sim0.65,\sim0.85]$ | $t\in[0.243,0.381]$, $x\in[0.693,0.860]$ |

エネルギー収支は両例とも機械精度で閉じる（相対 drift $1.3\times10^{-10}$ / $4.8\times10^{-10}$）。

### 7.3 収束テスト（論文が「実施した」と述べているもの）

`results/convergence/convergence.txt`。Example 1、$\varepsilon=5\times10^{-4}$ 固定:

| $\Delta x=\Delta t$ | $\Delta t/\varepsilon$ | 接触面積 | $\|\eta(T)-\eta_{\mathrm{ref}}\|_\infty$ | 備考 |
|---|---|---|---|---|
| $10^{-3}$ | 2.00 | 0.00099 | 8.50 | **不安定**（§4 (g)）|
| $5\times10^{-4}$ | 1.00 | 0.0494 | $5.5\times10^{-2}$ | **不安定** |
| $4\times10^{-4}$ | 0.80 | 0.0861 | $2.6\times10^{-2}$ | |
| $2\times10^{-4}$ | 0.40 | 0.1542 | $2.5\times10^{-3}$ | **論文設定** |
| $10^{-4}$ | 0.20 | 0.1561 | — | 参照解 |

$\varepsilon$ 掃引でも $\|\eta(T)-\eta_{\mathrm{ref}}\|_\infty$ は単調減少。Example 2 も同様の挙動。

---

## 8. リスク・注意点（計画時の予想と実際）

| 計画時のリスク | 実際 |
|---|---|
| 論文式 (b) の解釈違いで Ex.2 の形状が合わない | **現実になった**。paper-literal 版を必ず実行するという方針が効き、$E(0)$ 9 倍・$t=0$ 接触という定量的な証拠が取れた |
| 初期ステップ (d) の違いが接触開始時刻に影響し得る | 影響は $O(\Delta t)$ に留まった |
| 接触集合の見た目が閾値 (e) に敏感かもしれない | 面積比 1% 未満。**懸念しすぎだった** |
| MATLAB との丸め・線形ソルバの差 | 想定どおり無視できる（SPD 三重対角、drift $10^{-10}$）|
| 履歴の全保存で Ex.2 は ~100 MB | 実際 100 MB。`store_every` で対応。ただし `store_every` が総ステップ数を割り切らないと接触面積が過大になるバグを外部レビューで検出（`time_weights()` で修正、docs/notes.md §5）|
| （予想外）陽的ペナルティの安定条件 | $\Delta t/\varepsilon<1$ が必須。論文に記載がなく、収束テストの設計にも影響した（§4 (g)）|
| （予想外）初期データと境界条件の不整合 | 初期データを端点 1 に固定していたため $h\ne1$ で $\sim(1-h)^2/\Delta x$ の偽エネルギーが入り、「$h$ はほぼ無影響」という誤った結論を出した。$h+\text{shape}(x)$ 形式に修正し、不整合時は `solve()` が警告を出すようにした |

---

## 9. Phase 7 — 独自例（Example 3）と GIF アニメーション

論文の 2 例は、**弦全体がほぼ同時に落ちて一気に接触し、その後両側から縮む**という同じ型で、
$(t,x)$ 平面の接触集合はどちらも三角形状になる。そこで、検証済みのソルバーを使い、
**接触が別の起こり方をする例**を 1 つ作った。

### 9.1 設定（`--example 3`、論文にはない）

離散化パラメータは論文と同一（$l=1$, $\Delta x=\Delta t=1/5000$, $\alpha=0.01$, $\varepsilon=5\times10^{-4}$,
$h=1$、$\Delta t/\varepsilon=0.4$）、$T=0.7$（$M=3500$）。初期データ:

$$
\eta^0(x)=h+\tfrac32\sin(\pi x),\qquad
v^0(x)=-110\,\sin(\pi x)\cdot\tfrac12\Big(1-\tanh\tfrac{x-0.35}{0.1}\Big).
$$

- **狙い**: アーチ状の弦の**左半分だけ**に下向き速度を与える。左端側が先に障害物に当たって
  非弾性接触で貼り付き、その衝撃で放出された擾乱が右へ伝わって弦を順に寝かせていく。
- **初期データが両端の固定条件と完全に整合する**（$\eta^0(0)=\eta^0(1)=h$ かつ $v^0(0)=v^0(1)=0$）。
  論文の 2 例は $v^0$ が端点で $-50$ のまま不連続で、端点に初期層を作る。Example 3 にはそれがない。
- $|v^0|$ の最大は $x\approx0.26$ で 68.8、$x\ge0.7$ ではほぼ静止（$<0.1$）。

### 9.2 結果（`results/ex3/`）

| 項目 | 値 |
|---|---|
| 初回接触 | $t=0.0304$（$x\approx0.2$ 付近）|
| 接触終了 | $t=0.5754$ |
| 接触集合 | **連結成分 1 個**、$(t,x)$ 平面を**斜めに横切る帯**（$x\in[0.063,0.878]$）|
| 接触前線の速度 | 左縁 1.46、右縁 1.24（$t\in[0.1,0.5]$ の線形回帰）|
| 接触幅 | $t=0.1$ で 0.32、$t=0.3$ で 0.18、$t=0.5$ で 0.22 |
| 貫入深さ | $2.79\times10^{-2}$（$=55.8\,\varepsilon\approx|v^0|_{\max}\varepsilon$）|
| エネルギー | $E(0)=504.3\to E(T)=7.86$（粘性 155.7 / 接触 273.3 / 数値 67.4）、drift $8.0\times10^{-10}$ |
| 計算時間 | 0.66 秒 |

論文の 2 例が「三角形が縮んで 1 点で消える」のに対し、この例の接触集合は
**幅をほぼ保ったまま右へ平行移動する帯**になる。速度場（`velocity.png`）でも、
落下中（青）と接触中（赤 $\approx0$）を分ける直線的なフロントとして見える。
前線速度が波動部分の特性速度 1 をやや上回る（1.24–1.46）のは、粘性項 $\alpha\partial_{txx}\eta$ が
放物型で高周波の伝播速度に上限を与えないためと考えられる。

接触集合は依然として滑らかな境界を持つ 1 つの領域であり、Remark 2.2 の
「接触集合はかなり正則」という観察が、論文の 2 例とは異なる接触の起こり方でも成り立つことを示す。

### 9.3 アニメーション

```bash
uv run cdw animate --example 3 --frames 180 --fps 18 --dpi 85 --store-every 5
# -> results/ex3/animation.gif（2.0 MB, 180 フレーム, 765x544）
```

`src/contact_damped_wave/animation.py` は 3 パネルの GIF を書き出す（Pillow のみ使用、
ffmpeg 等の外部エンコーダは不要）:

1. **上**: 障害物の上の弦 $\eta(t,\cdot)$。接触中の部分（$\{\eta<0\}$）を太い黒線で強調し、
   初期形状を破線で残す。右上に現在時刻と接触区間の長さ。
2. **左下**: $(t,x)$ 平面の接触集合を、現在時刻まで**徐々に開示**（将来を白い矩形で覆い、
   1 フレームあたり $O(1)$ の更新にしている）。
3. **右下**: エネルギー $E(t)$ と累積散逸（粘性 + 接触）。接触の瞬間にエネルギーが落ちる。

`cdw animate --example 1 / 2` で論文の 2 例も同じ形式でアニメーション化できる
（`results/ex1/animation.gif`, `results/ex2/animation.gif`）。
