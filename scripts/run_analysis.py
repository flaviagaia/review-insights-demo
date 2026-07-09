"""EDA + validação de hipóteses. Gera as figuras usadas na apresentação.

Execução (a partir da raiz do repo):
    python scripts/run_analysis.py                          # amostra sintética
    python scripts/run_analysis.py --data-dir data/raw --sample-frac 0.13
Saída: reports/figures/*.png + reports/eda_summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_data
from src.nlp_pipeline import add_sentiment, aspect_sentiment, extract_topics, rank_reviewers

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLOR = "#4C72B0"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig -> {FIG_DIR / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="amostragem aleatória por chunks (ex.: 0.13)")
    ap.add_argument("--aspect-sample", type=int, default=120_000,
                    help="máx. de reviews para extração de aspectos (regex é O(n))")
    args = ap.parse_args()

    print("Carregando dados...")
    df = load_data(args.data_dir, nrows=args.nrows, sample_frac=args.sample_frac)
    lines = [f"# EDA — {len(df):,} reviews, {df['Title'].nunique()} livros, "
             f"{df['author'].nunique()} autores, {df['genre'].nunique()} gêneros\n"]

    # ---- 1. Distribuição de notas (viés positivo) -----------------------
    fig, ax = plt.subplots(figsize=(5, 3))
    df["review/score"].value_counts().sort_index().plot.bar(ax=ax, color=COLOR)
    ax.set_title("Distribuição de notas — viés positivo esconde críticas")
    ax.set_xlabel("Estrelas"); ax.set_ylabel("Reviews")
    save(fig, "01_dist_notas.png")
    pct_45 = (df["review/score"] >= 4).mean() * 100
    lines.append(f"- {pct_45:.0f}% das reviews têm 4-5★ (nota inflada → texto é o sinal).")

    # ---- 2. Long tail de volume por livro (H5) ---------------------------
    counts = df["Title"].value_counts().values
    cum = np.cumsum(counts) / counts.sum()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color=COLOR)
    ax.axhline(80, ls="--", c="gray", lw=0.8)
    ax.set_title("Concentração de reviews (curva de Pareto)")
    ax.set_xlabel("% dos livros"); ax.set_ylabel("% das reviews")
    save(fig, "02_pareto_livros.png")
    n80 = int(np.searchsorted(cum, 0.8)) + 1
    lines.append(f"- {n80} livros ({n80 / len(counts) * 100:.0f}%) concentram 80% das reviews "
                 f"→ análise manual gasta tempo na cauda longa.")

    # ---- 3. Sentimento do texto vs estrelas (H1/H3) ----------------------
    print("Treinando modelo de sentimento (supervisão fraca)...")
    df, model = add_sentiment(df)
    print(model.report_)
    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column="sentiment", by="review/score", ax=ax, grid=False)
    ax.set_title("Sentimento do TEXTO por nota — 3★ é ambíguo, o texto resolve")
    plt.suptitle("")
    ax.set_xlabel("Estrelas"); ax.set_ylabel("Sentimento (0-1)")
    save(fig, "03_sentimento_vs_nota.png")
    amb = df[df["review/score"] == 3]
    lines.append(f"- Reviews 3★: {len(amb):,} no total; o modelo separa "
                 f"{(amb['sentiment'] < 0.4).mean() * 100:.0f}% como negativas e "
                 f"{(amb['sentiment'] > 0.6).mean() * 100:.0f}% como positivas — "
                 f"informação invisível na nota.")

    # ---- 4. Performance por autor ----------------------------------------
    top_authors = (df[df["author"] != "Unknown"].groupby("author")
                   .agg(n=("review/score", "size"), nota=("review/score", "mean"),
                        sent=("sentiment", "mean"))
                   .query("n >= 30").sort_values("n", ascending=False).head(20)
                   .sort_values("sent"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_authors.index, top_authors["sent"], color=COLOR)
    ax.set_title("Sentimento médio por autor (≥30 reviews)")
    ax.set_xlabel("Sentimento médio do texto")
    save(fig, "04_sentimento_autor.png")

    # ---- 5. Aspectos por gênero (H4) --------------------------------------
    print("Extraindo aspectos...")
    asp_df = df if len(df) <= args.aspect_sample else df.sample(args.aspect_sample, random_state=42)
    asp = aspect_sentiment(asp_df)
    asp = asp.merge(df[["genre"]], left_on="review_idx", right_index=True)
    top_genres = df[df["genre"] != "Unknown"]["genre"].value_counts().head(8).index
    pivot = (asp[(asp["polarity"] != 0) & (asp["genre"].isin(top_genres))]
             .groupby(["genre", "aspect"])["polarity"].mean().unstack().round(2))
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Polaridade média por aspecto e gênero")
    fig.colorbar(im, shrink=0.8)
    save(fig, "05_aspectos_genero.png")
    lines.append("- Aspectos criticados variam por gênero (ex.: ritmo em ficção, "
                 "clareza/exemplos em não-ficção) → ação editorial direcionada.")

    # ---- 6. Reviewers relevantes (H2) --------------------------------------
    top = rank_reviewers(df, top_k=10)
    top.to_csv("reports/top_reviewers.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(top["profileName"].str[:28][::-1], top["relevance_score"][::-1], color=COLOR)
    ax.set_title("Top 10 leitores para entrevista (score de relevância)")
    save(fig, "06_top_reviewers.png")
    if df["helpful_votes"].sum() > 0:
        conc = df.groupby("User_id")["helpful_votes"].sum().sort_values(ascending=False)
        p10 = conc.head(max(1, len(conc) // 10)).sum() / max(conc.sum(), 1) * 100
        lines.append(f"- 10% dos usuários concentram {p10:.0f}% dos votos de utilidade "
                     f"→ shortlist objetiva de entrevistados.")
    else:
        conc = df.groupby("User_id")["review/text"].size().sort_values(ascending=False)
        p10 = conc.head(max(1, len(conc) // 10)).sum() / max(conc.sum(), 1) * 100
        lines.append(f"- Variante do dataset sem coluna de helpfulness: concentração medida "
                     f"por volume — 10% dos usuários escrevem {p10:.0f}% das reviews; "
                     f"ranking usa profundidade, produtividade e discriminação de nota.")

    # ---- 7. Tópicos ---------------------------------------------------------
    print("Extraindo tópicos (NMF)...")
    topics = extract_topics(df["review/text"], n_topics=6)
    lines.append("\n## Tópicos dominantes (NMF)\n")
    lines += [f"- Tópico {i + 1}: {', '.join(t)}" for i, t in enumerate(topics)]

    Path("reports/eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nOK — figuras em reports/figures/")


if __name__ == "__main__":
    main()
