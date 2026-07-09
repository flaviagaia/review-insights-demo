"""Estágio 4: figuras + sumário da EDA real, em partes com checkpoint.

Uso: python scripts/stage_figures.py {core|aspects|final}
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp_pipeline import aspect_sentiment, extract_topics, rank_reviewers

FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
LINES = Path("data/interim/eda_lines.json")
COLOR = "#4C72B0"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print("fig ->", name)


def put(key, line):
    d = json.loads(LINES.read_text()) if LINES.exists() else {}
    d[key] = line; LINES.write_text(json.dumps(d, ensure_ascii=False))


df = pd.read_parquet("data/interim/scored.parquet")
part = sys.argv[1]

if part == "core":
    put("head", f"# EDA (dados reais, amostra aleatória de {len(df):,} de 3M reviews) — "
                f"{df['Title'].nunique():,} livros, {df['author'].nunique():,} autores")
    fig, ax = plt.subplots(figsize=(5, 3))
    df["review/score"].value_counts().sort_index().plot.bar(ax=ax, color=COLOR)
    ax.set_title("Distribuição de notas — viés positivo esconde críticas")
    ax.set_xlabel("Estrelas"); ax.set_ylabel("Reviews")
    save(fig, "01_dist_notas.png")
    put("h1", f"- {(df['review/score'] >= 4).mean()*100:.0f}% das reviews têm 4-5★ "
              f"(nota média satura → o texto é o sinal).")

    counts = df["Title"].value_counts().values
    cum = np.cumsum(counts) / counts.sum()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color=COLOR)
    ax.axhline(80, ls="--", c="gray", lw=0.8)
    ax.set_title("Concentração de reviews (curva de Pareto)")
    ax.set_xlabel("% dos livros"); ax.set_ylabel("% das reviews")
    save(fig, "02_pareto_livros.png")
    n80 = int(np.searchsorted(cum, 0.8)) + 1
    put("h5", f"- {n80/len(counts)*100:.0f}% dos livros concentram 80% das reviews "
              f"→ esforço manual mal alocado na cauda longa.")

    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column="sentiment", by="review/score", ax=ax, grid=False)
    ax.set_title("Sentimento do TEXTO por nota — 3★ é ambíguo, o texto resolve")
    plt.suptitle(""); ax.set_xlabel("Estrelas"); ax.set_ylabel("Sentimento (0-1)")
    save(fig, "03_sentimento_vs_nota.png")
    amb = df[df["review/score"] == 3]
    put("h3", f"- Reviews 3★ ({len(amb):,}): o modelo separa "
              f"{(amb['sentiment'] < .4).mean()*100:.0f}% como negativas e "
              f"{(amb['sentiment'] > .6).mean()*100:.0f}% como positivas — "
              f"informação invisível na nota.")

    top = (df[df["author"] != "Unknown"].groupby("author")
           .agg(n=("review/score", "size"), sent=("sentiment", "mean"))
           .query("n >= 200").sort_values("n", ascending=False).head(18).sort_values("sent"))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(top.index, top["sent"], color=COLOR)
    ax.set_title("Sentimento médio por autor (autores mais avaliados)")
    ax.set_xlabel("Sentimento médio do texto")
    save(fig, "04_sentimento_autor.png")

elif part == "aspects":
    sample = df.sample(min(60_000, len(df)), random_state=42)
    asp = aspect_sentiment(sample)
    asp = asp.merge(df[["genre"]], left_on="review_idx", right_index=True)
    top_genres = df[df["genre"] != "Unknown"]["genre"].value_counts().head(8).index
    pivot = (asp[(asp["polarity"] != 0) & (asp["genre"].isin(top_genres))]
             .groupby(["genre", "aspect"])["polarity"].mean().unstack().round(2))
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Polaridade média por aspecto e gênero (top 8 gêneros)")
    fig.colorbar(im, shrink=0.8)
    save(fig, "05_aspectos_genero.png")
    put("h4", "- Aspectos criticados variam por gênero → ação editorial direcionada "
              "(ex.: ritmo em ficção; clareza/atualização em técnicos).")

elif part == "final":
    top = rank_reviewers(df, top_k=10)
    top.to_csv("reports/top_reviewers.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(top["profileName"].astype(str).str[:28][::-1],
            top["relevance_score"][::-1], color=COLOR)
    ax.set_title("Top 10 leitores para entrevista (score de relevância)")
    save(fig, "06_top_reviewers.png")
    conc = df.groupby("User_id")["review/text"].size().sort_values(ascending=False)
    p10 = conc.head(max(1, len(conc)//10)).sum() / conc.sum() * 100
    put("h2", f"- 10% dos usuários escrevem {p10:.0f}% das reviews; variante do dataset "
              f"sem coluna helpfulness → ranking usa profundidade, produtividade e "
              f"discriminação de nota.")

    topics = extract_topics(df["review/text"].sample(60_000, random_state=42), n_topics=6)
    put("topics", "\n## Tópicos dominantes (NMF)\n\n" +
        "\n".join(f"- Tópico {i+1}: {', '.join(t)}" for i, t in enumerate(topics)))

    d = json.loads(LINES.read_text())
    order = ["head", "h1", "h5", "h3", "h4", "h2", "topics"]
    Path("reports/eda_summary.md").write_text(
        "\n".join(d[k] for k in order if k in d) + "\n\nModelo de sentimento:\n" +
        Path("data/interim/sentiment_report.txt").read_text())
    print(Path("reports/eda_summary.md").read_text())
