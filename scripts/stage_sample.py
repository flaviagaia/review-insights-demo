"""Estágio 1: amostragem rápida do CSV real (pyarrow streaming) -> parquet.

Uso: python scripts/stage_sample.py <dir_dados> <frac> <saida.parquet>
"""
import sys

import numpy as np
import pyarrow.csv as pv
import pyarrow.parquet as pq
import pyarrow as pa

src, frac, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(42)

reader = pv.open_csv(
    f"{src}/Books_rating.csv",
    read_options=pv.ReadOptions(block_size=64 << 20),
)
batches, total = [], 0
for batch in reader:
    total += batch.num_rows
    mask = rng.random(batch.num_rows) < frac
    idx = np.nonzero(mask)[0]
    if len(idx):
        batches.append(batch.take(pa.array(idx)))
tbl = pa.Table.from_batches(batches)
pq.write_table(tbl, out, compression="zstd")
print(f"total={total:,} amostra={tbl.num_rows:,} -> {out}")
