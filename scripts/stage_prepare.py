"""Estágio 2: limpeza + merge com metadados -> clean.parquet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_parquet

df = load_parquet("data/interim/ratings_sample.parquet",
                  "data/raw")
print(f"{len(df):,} reviews limpas | {df['Title'].nunique():,} livros | "
      f"{df['author'].nunique():,} autores | {df['genre'].nunique():,} gêneros")
df.to_parquet("data/interim/clean.parquet", compression="zstd")
print("-> data/interim/clean.parquet")
