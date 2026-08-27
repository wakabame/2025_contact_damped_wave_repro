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
- [ ] Phase 1: ソルバー実装
- [ ] Phase 2: 検証テスト
- [ ] Phase 3: Example 1 再現（Fig. 2, 3）
- [ ] Phase 4: Example 2 再現（Fig. 4, 5）
- [ ] Phase 5–6: 追加解析・レポート

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

## 使い方（Phase 1 以降で有効になる予定のコマンド）

```bash
uv run cdw run --example 1 --out results/ex1     # Fig. 2, 3 相当を生成
uv run cdw run --example 2 --out results/ex2     # Fig. 4, 5 相当を生成
uv run cdw run --example 2 --initial paper-literal   # 論文本文の式そのままの初期データで実行
uv run python scripts/convergence_study.py       # Δt, Δx, ε の収束テスト
uv run pytest                                     # 検証テスト
uv run ruff check . && uv run ruff format .
```

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
src/contact_damped_wave/    ソルバー・初期データ・診断・描画・CLI（Phase 1 で実装）
tests/                      検証テスト（Phase 2）
scripts/                    一括実行・収束テスト用スクリプト
results/                    生成した図・データ（data/ は git 管理外）
paper/                      論文 PDF とページ画像（git 管理外）
```

## 参考文献

- B. Muha, S. Trifunović, *Analysis of an Inelastic Contact Problem for the Damped Wave Equation*,
  arXiv:2412.06185, <https://arxiv.org/abs/2412.06185>
