"""Estágio 3: sentimento por supervisão fraca com checkpoints.

Uso:
    python scripts/stage_sentiment.py fit          # treina em subamostra e salva modelo
    python scripts/stage_sentiment.py score <i> <n_slices>   # pontua a fatia i
    python scripts/stage_sentiment.py merge <n_slices>       # junta fatias
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp_pipeline import SentimentModel

CLEAN = "data/interim/clean.parquet"
MODEL = "data/interim/sentiment_model.joblib"

cmd = sys.argv[1]
if cmd == "fit":
    df = pd.read_parquet(CLEAN, columns=["review/score", "review/text"])
    train = df.sample(min(100_000, len(df)), random_state=42)
    model = SentimentModel().fit(train)
    joblib.dump(model, MODEL, compress=3)
    Path("data/interim/sentiment_report.txt").write_text(model.report_)
    print(model.report_)
elif cmd == "score":
    i, n = int(sys.argv[2]), int(sys.argv[3])
    model = joblib.load(MODEL)
    df = pd.read_parquet(CLEAN, columns=["review/text"])
    sl = df.iloc[i::n]
    scores = model.score(sl["review/text"])
    np.save(f"data/interim/sent_{i}.npy", scores)
    print(f"fatia {i}/{n}: {len(scores):,} pontuadas")
elif cmd == "merge":
    n = int(sys.argv[2])
    df = pd.read_parquet(CLEAN)
    sent = np.empty(len(df))
    for i in range(n):
        sent[i::n] = np.load(f"data/interim/sent_{i}.npy")
    df["sentiment"] = sent
    df["sentiment_label"] = pd.cut(df["sentiment"], [0, .4, .6, 1.0],
                                   labels=["negativo", "neutro", "positivo"])
    df.to_parquet("data/interim/scored.parquet", compression="zstd")
    print(f"-> data/interim/scored.parquet ({len(df):,} linhas)")
