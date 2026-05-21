"""Generate manifests of medoid building IDs for parquet downloads.

Reads data/representative_buildings.parquet (the 2,541 stratified medoids)
and writes:
  data/medoid_bldg_ids_PGE.txt   (one ID per line — for download tools)
  data/medoid_bldg_ids_SCE.txt
  data/medoid_bldg_ids_SDGE.txt
  data/medoid_bldg_ids.csv       (combined, with utility column)

Use case: only the ~2,541 medoid `<bldg_id>-0.parquet` (baseline) and
`<bldg_id>-11.parquet` (Upgrade 11) files need to be downloaded from
Google Drive / cloud storage, not the full ResStock CA population
(~57,394 files). Per utility:
    PGE   1,378 medoids
    SCE     842
    SDGE    321
Total: ~5,082 parquets needed (baseline + upgrade11), vs ~115K otherwise.

Re-run this script after regenerating representative_buildings.parquet
if the medoid set changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


def main():
    med_path = config.DATA_DIR / "representative_buildings.parquet"
    if not med_path.exists():
        sys.exit(
            f"Medoid parquet not found at {med_path}; run "
            f"`python -m src.representative_buildings` first.")

    df = pd.read_parquet(med_path, columns=["bldg_id", "utility"])
    df["bldg_id"] = df["bldg_id"].astype(int)

    # Combined CSV
    combined_path = config.DATA_DIR / "medoid_bldg_ids.csv"
    df.sort_values(["utility", "bldg_id"]).to_csv(
        combined_path, index=False)
    print(f"Wrote combined: {combined_path}")

    # Per-utility plain-text lists (one bldg_id per line)
    for u in sorted(df["utility"].unique()):
        ids = sorted(df[df["utility"] == u]["bldg_id"].tolist())
        out = config.DATA_DIR / f"medoid_bldg_ids_{u}.txt"
        out.write_text("\n".join(str(i) for i in ids) + "\n")
        print(f"  {u}: {len(ids):>5,} ids -> {out}")

    print(f"\nTotal medoids: {len(df):,}")
    print(f"Files needed per medoid: 2 (baseline + upgrade11)")
    print(f"Total parquet files to download: {len(df) * 2:,}")


if __name__ == "__main__":
    main()
