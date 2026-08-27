# contact-damped-wave

arXiv:2412.06185 — B. Muha, S. Trifunović,
*Analysis of an Inelastic Contact Problem for the Damped Wave Equation* (v2, 2025-05-15)
の数値実験（Section 6, Example 1 / 2, Figures 2–5）を **Python + uv** で再現するプロジェクトです。

対象は、剛体障害物 $y=0$ の上で振動する 1 次元粘弾性弦の非弾性接触問題

$$
\partial_{tt}\eta-\alpha\,\partial_{txx}\eta-\partial_{xx}\eta
=\frac1\varepsilon\,\chi_{\{\eta<0\}}(\partial_t\eta)^-,\qquad \eta(t,0)=\eta(t,l)=h,
$$

を論文と同じ差分スキーム（空間 2 階中心差分・粘性/弾性項は陰的・ペナルティ項は陽的）で解き、
解のスナップショット、$(t,x)$ 平面上の接触集合、速度場を描画します。

計画の詳細・論文の転記・記述の不整合と採用方針は [`plan.md`](plan.md) を参照してください。

## 現在の状態

- [x] Phase 0: 環境構築（uv, 依存関係, plan.md, README）
- [x] Phase 1: ソルバー実装（`params` / `initial_data` / `solver` / `diagnostics` / `plotting` / `cli`）
- [x] Phase 2: 検証テスト（45 件。解析解との 1 次収束、エネルギー恒等式、対称性、収束テスト）
- [x] Phase 3: Example 1 再現（Fig. 2, 3）
- [x] Phase 4: Example 2 再現（Fig. 4, 5）
- [x] 外部レビュー（codex）の指摘 7 件を修正（[docs/notes.md](docs/notes.md) §5）
- [ ] Phase 5–6: 追加解析（接触散逸の局在）・図の並列比較

### 再現結果の要約

| | 論文 | 本再現 |
|---|---|---|
| **Ex.1** 初回接触 | ≈0.02 | 0.0236 |
| **Ex.1** 接触集合 | $x=0.5$ 対称の三角形、左縁に 10 個の波打ち | 同一（対称性は $1.7\times10^{-12}$）|
| **Ex.1** 接触消滅 | ≈0.26 | 0.299 — **唯一の差異**（[docs/notes.md](docs/notes.md) §3）|
| **Ex.2** 連結成分数 | 2 | 2 ✓ |
| **Ex.2** 大成分 | $t\in[\sim0.02,\sim0.26]$, $x\in[\sim0.1,\sim0.6]$ | $t\in[0.026,0.261]$, $x\in[0.032,0.594]$ ✓ |
| **Ex.2** 小成分 | $t\in[\sim0.25,\sim0.37]$, $x\in[\sim0.65,\sim0.85]$ | $t\in[0.243,0.381]$, $x\in[0.693,0.860]$ ✓ |

スナップショット（Fig. 2 / Fig. 4）は両例とも 6 枚すべて目視一致。
離散エネルギー収支は機械精度で閉じる（drift $\sim10^{-10}$）。計算時間は論文設定で 0.3–0.5 秒。

**論文記述の重要な問題を 2 点確認**（詳細は [docs/notes.md](docs/notes.md)）:

1. Example 2 の初期データの式は原文のままだと不連続（$x=0.8$ で 1.202 のジャンプ）で
   $\eta^0(0)=0$ となり Theorem 2.1 の仮定に反し、$E(0)$ が 9 倍になって Fig. 4 を再現しない。
   図に整合する式を既定とした（`--initial paper-literal` で原文式も実行可）。
2. 論文に記載のない**陽的ペナルティの安定条件 $\Delta t/\varepsilon<1$** が存在する。
   論文設定は $0.4$ で安全側だが、$\ge2$ ではエネルギーが増加して破綻する。

## セットアップ

[uv](https://docs.astral.sh/uv/) が必要です。Python 3.12 は uv が自動で用意します。

```bash
git clone <this-repo> && cd 2025_contact_damped_wave_repro
uv sync            # .venv を作成し、numpy / scipy / matplotlib と dev ツールを導入
uv run python -c "import contact_damped_wave, numpy, scipy, matplotlib; print('ok')"
```

論文 PDF は git 管理外です。参照する場合は次で取得してください。

```bash
mkdir -p paper && curl -L -o paper/2412.06185v2.pdf https://arxiv.org/pdf/2412.06185v2
```

## 使い方

```bash
uv run python scripts/run_examples.py            # 両例＋原文式版をまとめて実行
uv run cdw run --example 1                       # Fig. 2, 3 相当を results/ex1 に生成
uv run cdw run --example 2                       # Fig. 4, 5 相当を results/ex2 に生成
uv run cdw run --example 2 --initial paper-literal --out results/ex2_paper_literal
uv run python scripts/convergence_study.py       # Δt, Δx, ε の収束テスト
uv run pytest                                     # 検証テスト（45 件、1 秒未満）
uv run ruff check . && uv run ruff format .
```

主なオプション（`uv run cdw run --help`）:

| オプション | 既定 | 意味 |
|---|---|---|
| `--dx --dt --eps --alpha --final-time --h` | 論文値 | パラメータの上書き |
| `--initial {figure,paper-literal}` | `figure` | Ex.2 の初期データ（plan.md 項目 (b)）|
| `--initial-step {backward,forward}` | `backward` | 三段階漸化式の開始法（項目 (d)）|
| `--contact-mode {negative,threshold}` | `negative` | 接触集合の判定（項目 (e)）|
| `--connectivity {1,2}` | 2 | 連結成分の近傍（2 = 8 近傍）|
| `--store-every` | 1 | スナップショットの間引き |

生成物: `fig{2,4}_snapshots.png`, `fig{3,5}_contact_set.png`, `fig{3,5}_velocity.png`,
`energy.png`, `summary.txt`、およびパラメータを名前に含む `.npz`（すべて `--out` 配下）。

## 論文の設定（共通）

| パラメータ | 値 |
|-----------|----|
| $l$ | 1 |
| $\Delta t=\Delta x$ | $1/5000$ |
| $\alpha$（粘弾性係数） | 0.01 |
| $\varepsilon$（ペナルティ） | 0.0005 |
| $h$（端点の高さ） | 1（図より） |
| Example 1 | $T=0.3$, $\eta^0=1+\tfrac12\sin^2(10\pi x)$, $v^0=-50$ |
| Example 2 | $T=0.5$, 区分的初期データ（`plan.md` §3 参照）, $v^0=-50$ ($x<0.6$), $-0.5$ ($x\ge0.6$) |

## ディレクトリ

```
plan.md                     再現計画（論文の転記・不整合・フェーズ）
docs/notes.md               再現ノート（スキームの整理・不整合の裏付け・結果と差異）
src/contact_damped_wave/
  params.py                 Params データクラスと論文設定 EXAMPLE1 / EXAMPLE2
  initial_data.py           両例の初期データ（Ex.2 は figure / paper-literal の 2 種）
  solver.py                 三重対角陰解法ソルバーと厳密な離散エネルギー収支
  diagnostics.py            接触集合・連結成分・貫入深さ・エネルギー収支
  plotting.py               Fig. 2–5 相当の描画
  cli.py                    `cdw run` コマンド
tests/                      検証テスト 38 件
scripts/                    run_examples.py / convergence_study.py
results/                    生成した図・要約（data/ は git 管理外）
paper/                      論文 PDF とページ画像（git 管理外）
```

## 参考文献

- B. Muha, S. Trifunović, *Analysis of an Inelastic Contact Problem for the Damped Wave Equation*,
  arXiv:2412.06185, <https://arxiv.org/abs/2412.06185>
