"""Download medoid building parquets from OEDI's public S3 bucket.

Reads data/medoid_bldg_ids.csv (the 2,541 stratified medoids) and pulls
only those specific buildings from OEDI ResStock 2024 release 2.
Default target is Upgrade 11 (whole-home HP electrification); pass
--upgrade 0 for baseline parquets.

OEDI S3 layout (public, no auth needed; uses UNSIGNED boto3):
  s3://oedi-data-lake/nrel-pds-building-stock/...
      timeseries_individual_buildings/by_state/
        upgrade=<N>/state=CA/<bldg_id>-<N>.parquet

Local layout (matches what EE's bill.py expects):
  <out_dir>/Baseline_<UTILITY>/<bldg_id>-0.parquet         (--upgrade 0)
  <out_dir>/Upgrade11_<UTILITY>/<bldg_id>-11.parquet       (--upgrade 11)

Where <out_dir> defaults to config.PARQUET_ROOT (= EE_PARQUET_DIR env
var if set, else the EE repo root). Override with --out-dir to land
directly on a Drive mount, e.g.:

    python scripts/download_medoid_parquets.py --upgrade 11 \\
        --out-dir /Volumes/GoogleDrive/MyDrive/ee_parquets

The downloader is:
  - Idempotent (skips files that already exist and are non-empty)
  - Parallel (default 10 workers; --max-workers to tune)
  - Tolerant (one missing-on-S3 building doesn't abort the run)
  - Restartable (rerunning picks up only what's missing)

Requirements:
  pip install boto3
  (boto3 is NOT in requirements.txt — install separately, this script
   is a download utility outside the core analysis pipeline.)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


OEDI_BUCKET = "oedi-data-lake"
OEDI_PREFIX = (
    "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
    "/2024/resstock_amy2018_release_2/timeseries_individual_buildings"
    "/by_state"
)


def s3_key(upgrade: int, bldg_id: int) -> str:
    """OEDI S3 key for one (upgrade, bldg_id) on CA."""
    return (f"{OEDI_PREFIX}/upgrade={upgrade}/state=CA/"
            f"{bldg_id}-{upgrade}.parquet")


def download_one(s3_client, bucket: str, key: str,
                 local_path: Path) -> str:
    """Download one parquet. Returns 'ok' / 'skip' / 'fail:<msg>'."""
    if local_path.exists() and local_path.stat().st_size > 0:
        return "skip"
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        return "ok"
    except Exception as e:
        # Partial file from a failed download would block resume;
        # clean up so a rerun can retry.
        try:
            if local_path.exists():
                local_path.unlink()
        except OSError:
            pass
        return f"fail:{type(e).__name__}: {e}"


def download_utility(s3_client, df: pd.DataFrame, utility: str,
                      upgrade: int, out_root: Path,
                      max_workers: int, limit: int | None) -> dict:
    """Download all medoid parquets for one utility."""
    ids = sorted(df[df["utility"] == utility]["bldg_id"].astype(int).tolist())
    if limit:
        ids = ids[:limit]
    if not ids:
        return {"utility": utility, "ok": 0, "skip": 0, "fail": 0,
                "total": 0, "failures": []}

    folder = "Baseline_" if upgrade == 0 else f"Upgrade{upgrade}_"
    local_dir = out_root / f"{folder}{utility}"
    suffix = f"-{upgrade}.parquet"
    print(f"\n{utility}: {len(ids):,} medoids -> {local_dir}")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_id = {
            ex.submit(download_one, s3_client, OEDI_BUCKET,
                      s3_key(upgrade, bid),
                      local_dir / f"{bid}{suffix}"): bid
            for bid in ids
        }
        for i, fut in enumerate(as_completed(future_to_id), 1):
            bid = future_to_id[fut]
            result = fut.result()
            if result == "ok":
                counts["ok"] += 1
            elif result == "skip":
                counts["skip"] += 1
            else:
                counts["fail"] += 1
                failures.append((bid, result))
            if i % 50 == 0 or i == len(ids):
                print(f"  [{i:>5,}/{len(ids):,}] "
                      f"ok={counts['ok']} skip={counts['skip']} "
                      f"fail={counts['fail']}")

    if failures and len(failures) <= 5:
        for bid, msg in failures:
            print(f"    FAIL bldg {bid}: {msg}")
    elif failures:
        print(f"    {len(failures)} failures; first 5:")
        for bid, msg in failures[:5]:
            print(f"      bldg {bid}: {msg}")
    return {"utility": utility, "total": len(ids), **counts,
            "failures": failures}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None,
                    help="path to medoid_bldg_ids.csv "
                         "(default: data/medoid_bldg_ids.csv)")
    ap.add_argument("--upgrade", type=int, choices=[0, 11], default=11,
                    help="0 = baseline, 11 = whole-home HP (default 11)")
    ap.add_argument("--utilities", nargs="+",
                    default=["PGE", "SCE", "SDGE"],
                    choices=["PGE", "SCE", "SDGE"])
    ap.add_argument("--out-dir", default=None,
                    help="parquet root (default: EE_PARQUET_DIR env var "
                         "if set, else config.PARQUET_ROOT)")
    ap.add_argument("--max-workers", type=int, default=10,
                    help="parallel downloads (default 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="testing: only fetch first N medoids per utility")
    args = ap.parse_args()

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config as BotoConfig
    except ImportError:
        sys.exit("Need boto3: `pip install boto3`")

    manifest_path = Path(args.manifest) if args.manifest else (
        config.DATA_DIR / "medoid_bldg_ids.csv")
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\n"
                 f"Run `python scripts/list_medoid_bldg_ids.py` first.")
    df = pd.read_csv(manifest_path)

    out_root = Path(args.out_dir) if args.out_dir else config.PARQUET_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client(
        "s3", config=BotoConfig(signature_version=UNSIGNED))

    summary = []
    for u in args.utilities:
        summary.append(download_utility(
            s3_client, df, u, args.upgrade, out_root,
            args.max_workers, args.limit))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'utility':<8} {'total':>7} {'ok':>7} {'skip':>7} {'fail':>7}")
    grand = {"total": 0, "ok": 0, "skip": 0, "fail": 0}
    for s in summary:
        print(f"{s['utility']:<8} {s['total']:>7,} {s['ok']:>7,} "
              f"{s['skip']:>7,} {s['fail']:>7,}")
        for k in grand:
            grand[k] += s[k]
    print("-" * 38)
    print(f"{'TOTAL':<8} {grand['total']:>7,} {grand['ok']:>7,} "
          f"{grand['skip']:>7,} {grand['fail']:>7,}")
    if grand["fail"]:
        print(f"\n{grand['fail']} files failed. Rerun the script to retry "
              f"(successful files will be skipped).")
        sys.exit(1)


if __name__ == "__main__":
    main()
