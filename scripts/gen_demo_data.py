"""Pré-computa os dados do MVP em JSON para a demo HTML standalone."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import hashlib

import pandas as pd


from src.nlp_pipeline import aspect_sentiment, rank_reviewers
from src.summarizer import summarize_entity

import os
# executa a partir da raiz do repo

# Dados REAIS já pontuados pelo pipeline (amostra aleatória de 3M reviews)
df = pd.read_parquet("data/interim/scored.parquet")



def pseudo(user_id: str) -> str:
    """Pseudonimização (LGPD): demo pública nunca exibe nomes reais de usuários."""
    return "Leitor " + hashlib.sha256(str(user_id).encode()).hexdigest()[:6].upper()

def entity_payload(sub, name, kind):
    asp = aspect_sentiment(sub)
    aspects = []
    if not asp.empty:
        agg = (asp[asp["polarity"] != 0].groupby("aspect")
               .agg(n=("polarity", "size"), pol=("polarity", "mean")).reset_index()
               .sort_values("n", ascending=False).head(10))
        aspects = [{"a": r["aspect"], "n": int(r["n"]), "p": round(float(r["pol"]), 2)}
                   for _, r in agg.iterrows()]
    yearly = (sub.groupby("review_year")["sentiment"].mean().dropna().reset_index())
    hist = sub["review/score"].value_counts().sort_index()
    rv = rank_reviewers(sub if len(sub) > 200 else df, top_k=6)
    rv["profileName"] = rv["User_id"].map(pseudo)
    return {
        "name": name, "kind": kind,
        "n": int(len(sub)),
        "score": round(float(sub["review/score"].mean()), 2),
        "sent": round(float(sub["sentiment"].mean()), 3),
        "users": int(sub["User_id"].nunique()),
        "hist": {int(k): int(v) for k, v in hist.items()},
        "yearly": [[int(r["review_year"]), round(float(r["sentiment"]), 3)]
                   for _, r in yearly.iterrows()],
        "aspects": aspects,
        "summary": summarize_entity(sub, f"{kind} '{name}'", "mock"),
        "reviewers": [
            {"name": r["profileName"], "n": int(r["n_reviews"]),
             "votes": int(r["total_helpful_votes"]), "len": int(r["avg_len"]),
             "score": round(float(r["avg_score"]), 2), "rel": float(r["relevance_score"])}
            for _, r in rv.iterrows()],
    }

entities = []
authors = df[df["author"] != "Unknown"]["author"].value_counts().head(15).index
for a in authors:
    entities.append(entity_payload(df[df["author"] == a], a, "Autor"))
genres = df[df["genre"] != "Unknown"]["genre"].value_counts().head(10).index
for g in genres:
    entities.append(entity_payload(df[df["genre"] == g], g, "Gênero"))
for t in df["Title"].value_counts().head(10).index:
    entities.append(entity_payload(df[df["Title"] == t], t, "Livro"))

qa_corpus = [
    {"a": r["author"], "t": str(r["Title"])[:80], "g": r["genre"],
     "s": int(r["review/score"]), "x": str(r["review/text"])[:260]}
    for _, r in df.sample(8000, random_state=42).iterrows()
]

data = {"entities": entities, "qa": qa_corpus,
        "meta": {"reviews": int(len(df)), "books": int(df["Title"].nunique()),
                 "authors": int(df["author"].nunique()),
                 "genres": int(df["genre"].nunique())}}
out = Path("../demo_data.json")
out.write_text(json.dumps(data, ensure_ascii=False))
print(f"OK: {out} ({out.stat().st_size/1e6:.1f} MB, {len(entities)} entidades, {len(qa_corpus)} reviews)")
