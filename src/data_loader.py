"""Carga e limpeza dos dados (Kaggle Amazon Books Reviews ou amostra sintética).

Detecta automaticamente data/raw/ (dados reais do Kaggle) e cai para
data/sample/ (amostra sintética) se os dados reais não estiverem presentes.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/sample")


def _parse_list_col(value) -> str:
    """Colunas 'authors' e 'categories' vêm como string de lista: "['X']" -> "X"."""
    if pd.isna(value):
        return "Unknown"
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return str(value)


def _parse_helpfulness(value) -> tuple[int, int]:
    """'7/10' -> (7, 10)."""
    if isinstance(value, str) and re.fullmatch(r"\d+/\d+", value):
        a, b = value.split("/")
        return int(a), int(b)
    return 0, 0


# Variantes de schema encontradas em downloads do dataset (robustez de ingest)
COLUMN_ALIASES = {
    "score": "review/score", "time": "review/time",
    "summary": "review/summary", "text": "review/text",
    "helpfulness": "review/helpfulness",
}


def load_data(data_dir: str | Path | None = None, nrows: int | None = None,
              sample_frac: float | None = None, seed: int = 42) -> pd.DataFrame:
    """Carrega ratings + metadados e devolve um DataFrame unificado e limpo.

    Args:
        data_dir: pasta com Books_rating.csv e books_data.csv. Se None,
            usa data/raw/ (real) ou data/sample/ (sintética).
        nrows: limite de linhas de reviews (útil para o CSV real de 2.7GB).
        sample_frac: amostragem aleatória por chunks (ex.: 0.13 ≈ 390k linhas
            do CSV completo) — evita o viés de posição do `nrows` e cabe em RAM.
    """
    if data_dir is None:
        data_dir = RAW_DIR if (RAW_DIR / "Books_rating.csv").exists() else SAMPLE_DIR
    data_dir = Path(data_dir)

    if sample_frac:
        chunks = [c.sample(frac=sample_frac, random_state=seed)
                  for c in pd.read_csv(data_dir / "Books_rating.csv", chunksize=250_000)]
        ratings = pd.concat(chunks, ignore_index=True)
    else:
        ratings = pd.read_csv(data_dir / "Books_rating.csv", nrows=nrows)
    ratings = ratings.rename(columns=COLUMN_ALIASES)
    if "review/helpfulness" not in ratings.columns:
        ratings["review/helpfulness"] = np.nan  # variante do dataset sem a coluna

    books = pd.read_csv(
        data_dir / "books_data.csv",
        usecols=["Title", "authors", "categories", "publisher", "publishedDate"],
    )
    return _clean_merge(ratings, books)


def load_parquet(parquet_path: str | Path, data_dir: str | Path) -> pd.DataFrame:
    """Carrega um checkpoint parquet de ratings + books_data.csv do data_dir."""
    ratings = pd.read_parquet(parquet_path).rename(columns=COLUMN_ALIASES)
    if "review/helpfulness" not in ratings.columns:
        ratings["review/helpfulness"] = np.nan
    books = pd.read_csv(
        Path(data_dir) / "books_data.csv",
        usecols=["Title", "authors", "categories", "publisher", "publishedDate"],
    )
    return _clean_merge(ratings, books)


def _clean_merge(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    # -- limpeza básica --------------------------------------------------
    ratings = ratings.dropna(subset=["Title", "review/text"]).copy()
    ratings["review/text"] = (
        ratings["review/text"].astype(str)
        .str.replace(r"<[^>]{1,40}>", " ", regex=True)   # tags HTML (<br />, <p>...)
        .str.replace("&quot;", '"').str.replace("&amp;", "&")
        .str.replace("&#39;", "'").str.replace("&gt;", ">").str.replace("&lt;", "<")
        .str.strip()
    )
    ratings = ratings[ratings["review/text"].str.len() >= 10]
    ratings = ratings.drop_duplicates(subset=["Title", "User_id", "review/text"])

    # títulos duplicados em books_data (edições múltiplas) explodiriam o merge
    books = books.drop_duplicates(subset=["Title"]).copy()
    books["author"] = books["authors"].apply(_parse_list_col)
    books["genre"] = books["categories"].apply(_parse_list_col)

    df = ratings.merge(
        books[["Title", "author", "genre", "publisher", "publishedDate"]],
        on="Title", how="left",
    )
    df["author"] = df["author"].fillna("Unknown")
    df["genre"] = df["genre"].fillna("Unknown")

    # -- features derivadas ----------------------------------------------
    helpful = df["review/helpfulness"].apply(_parse_helpfulness)
    df["helpful_votes"] = helpful.str[0]
    df["total_votes"] = helpful.str[1]
    df["helpful_ratio"] = np.where(
        df["total_votes"] > 0, df["helpful_votes"] / df["total_votes"], np.nan
    )
    df["review_date"] = pd.to_datetime(df["review/time"], unit="s", errors="coerce")
    df["review_year"] = df["review_date"].dt.year
    df["review_len"] = df["review/text"].str.count(" ").add(1)  # ~palavras, O(n) leve

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df[["Title", "author", "genre", "review/score", "helpful_ratio"]].head())
