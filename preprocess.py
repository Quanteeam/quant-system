"""Preprocess raw Sharadar ZIP exports into a local parquet bundle.

Input layout (--raw-dir):
    SHARADAR_SEP_*.zip      equity daily prices
    SHARADAR_SFP_*.zip      fund/ETF prices  (SPY, QQQ extracted from here)
    SHARADAR_SF1_*.zip      fundamental financials
    SHARADAR_TICKERS_*.zip  ticker metadata

Output layout (--out-dir):
    tickers.parquet
    sf1.parquet
    sep/
        ticker=AAPL/data.parquet
        ticker=MSFT/data.parquet
        ticker=SPY/data.parquet   <- merged from SFP
        ...

Usage:
    python preprocess.py \\
        --raw-dir "C:/Users/womin/OneDrive/바탕 화면/quant_data" \\
        --out-dir "C:/Users/womin/quant_data"
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Column specs
# ---------------------------------------------------------------------------

# Columns kept in each sep/ticker=X/data.parquet  (no "ticker" column here;
# it is inferred from the directory name by local.py)
SEP_FILE_COLS = ["date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj", "lastupdated"]

# All available SEP/SFP columns read from CSV (ticker needed for groupby)
SEP_CSV_COLS = ["ticker"] + SEP_FILE_COLS

# Benchmark ETFs to pull from SFP and merge into sep/
BENCHMARK_TICKERS = ["SPY", "QQQ"]
BENCHMARK_WARN_START = pd.Timestamp("2015-01-01")
MIN_FREE_GB = 5.0

# TICKERS columns to keep
TICKERS_COLS = [
    "table", "permaticker", "ticker", "name", "exchange", "isdelisted",
    "category", "cusips", "siccode", "sicsector", "sector", "industry",
    "firstpricedate", "lastpricedate",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_zip(raw_dir: Path, prefix: str) -> Path:
    matches = list(raw_dir.glob(f"{prefix}*.zip"))
    if not matches:
        raise FileNotFoundError(f"No ZIP file found matching prefix '{prefix}' in {raw_dir}")
    # Pick the most recently modified file; warn if multiple exist so the user
    # knows which one was selected.
    chosen = max(matches, key=lambda p: p.stat().st_mtime)
    if len(matches) > 1:
        names = sorted(p.name for p in matches)
        print(f"  [WARN] Multiple ZIPs match '{prefix}': {names}")
        print(f"         Using newest by mtime: {chosen.name}")
    return chosen


def _read_zip_csv(zip_path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        csv_name = z.namelist()[0]
        # Read header first to intersect requested columns with what actually exists.
        with z.open(csv_name) as f:
            actual_cols = pd.read_csv(f, nrows=0).columns.tolist()
        if usecols is not None:
            missing = [c for c in usecols if c not in actual_cols]
            if missing:
                print(f"  [WARN] {zip_path.name}: columns not found, skipping: {missing}")
            usecols = [c for c in usecols if c in actual_cols]
            if not usecols:
                raise ValueError(f"{zip_path.name}: none of the requested columns exist")
        with z.open(csv_name) as f:
            return pd.read_csv(f, usecols=usecols, low_memory=False)


def _cast_sep(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same dtypes as the existing processed data."""
    price_cols = ["open", "high", "low", "close", "closeadj", "closeunadj"]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    for col in ("date", "lastupdated"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _warn_disk_space(out_dir: Path) -> None:
    free_gb = shutil.disk_usage(out_dir).free / 1024 ** 3
    if free_gb < MIN_FREE_GB:
        print(f"  [WARN] Only {free_gb:.1f} GB free in {out_dir}. Recommend {MIN_FREE_GB:.0f}+ GB.")


def _spot_check_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    if df.empty:
        raise ValueError(f"Output parquet is empty: {path}")
    return df


# ---------------------------------------------------------------------------
# Step 1: tickers.parquet
# ---------------------------------------------------------------------------

def process_tickers(raw_dir: Path, out_dir: Path) -> None:
    print("[1/3] Processing TICKERS...")
    zip_path = _find_zip(raw_dir, "SHARADAR_TICKERS")
    df = _read_zip_csv(zip_path)

    # Keep only columns that exist in this version of the dataset
    keep = [c for c in TICKERS_COLS if c in df.columns]
    df = df[keep].copy()

    for col in ("firstpricedate", "lastpricedate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    out_path = out_dir / "tickers.parquet"
    df.to_parquet(out_path, index=False)
    _spot_check_parquet(out_path, columns=[c for c in ["ticker", "table"] if c in df.columns])
    print(f"  -> {out_path}  ({len(df):,} rows)")
    if "table" in df.columns:
        print(f"     tables: {df['table'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Step 2: sep/ partitioned by ticker
# ---------------------------------------------------------------------------

def process_sep(raw_dir: Path, out_dir: Path) -> None:
    print("[2/3] Processing SEP + SFP (SPY/QQQ)...")

    # --- read SEP ---
    sep_zip = _find_zip(raw_dir, "SHARADAR_SEP")
    print(f"  Reading {sep_zip.name} ...")
    sep_df = _read_zip_csv(sep_zip, usecols=SEP_CSV_COLS)
    print(f"  SEP: {len(sep_df):,} rows, {sep_df['ticker'].nunique():,} tickers")

    # --- read SPY/QQQ from SFP ---
    sfp_zip = _find_zip(raw_dir, "SHARADAR_SFP")
    print(f"  Reading {sfp_zip.name} (SPY/QQQ only) ...")
    sfp_df = _read_zip_csv(sfp_zip, usecols=SEP_CSV_COLS)
    bench_df = sfp_df[sfp_df["ticker"].isin(BENCHMARK_TICKERS)].copy()
    del sfp_df
    print(f"  SFP benchmark rows: {len(bench_df):,} ({bench_df['ticker'].value_counts().to_dict()})")

    # --- merge; SEP takes precedence if ticker appears in both ---
    sep_tickers = set(sep_df["ticker"].unique())
    bench_df = bench_df[~bench_df["ticker"].isin(sep_tickers)]
    combined = pd.concat([sep_df, bench_df], ignore_index=True)
    combined = _cast_sep(combined)
    sort_cols = [c for c in ["ticker", "date", "lastupdated"] if c in combined.columns]
    before_dedup = len(combined)
    combined = combined.sort_values(sort_cols)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    dropped = before_dedup - len(combined)
    if dropped:
        print(f"  [WARN] Dropped {dropped:,} duplicate (ticker, date) rows using latest lastupdated")
    print(f"  Combined: {len(combined):,} rows, {combined['ticker'].nunique():,} tickers")

    for bench in BENCHMARK_TICKERS:
        bench_dates = combined.loc[combined["ticker"].eq(bench), "date"]
        if bench_dates.empty:
            print(f"  [WARN] {bench} benchmark data is missing")
            continue
        min_date = bench_dates.min()
        if min_date > BENCHMARK_WARN_START:
            print(
                f"  [WARN] {bench} data starts {min_date.date()} - "
                "backtests before this date will have no benchmark curve"
            )

    # --- write per-ticker parquet files into a temp dir, then swap ---
    # Writing to sep_tmp first prevents stale tickers from a previous run
    # surviving in sep/ if the ticker list or benchmark set changes.
    sep_final = out_dir / "sep"
    sep_tmp = out_dir / "sep_tmp"
    if sep_tmp.exists():
        shutil.rmtree(sep_tmp)
    sep_tmp.mkdir(parents=True)

    n_written = 0
    for ticker, group in combined.groupby("ticker", sort=False):
        ticker_dir = sep_tmp / f"ticker={ticker}"
        ticker_dir.mkdir(exist_ok=True)
        # Drop ticker column; local.py infers it from directory name
        drop_cols = [c for c in ["ticker", "lastupdated"] if c in group.columns]
        out_df = group.drop(columns=drop_cols).reset_index(drop=True)
        out_df.to_parquet(ticker_dir / "data.parquet", index=False)
        n_written += 1
        if n_written % 2000 == 0:
            print(f"  ... {n_written} tickers written")

    # Swap: remove old sep/, rename sep_tmp -> sep/
    if sep_final.exists():
        shutil.rmtree(sep_final)
    sep_tmp.rename(sep_final)
    sample_ticker = next(sep_final.iterdir(), None)
    if sample_ticker is None:
        raise ValueError(f"No ticker parquet files were written under {sep_final}")
    spot_check = _spot_check_parquet(sample_ticker / "data.parquet")
    if "date" not in spot_check.columns:
        raise ValueError(f"Missing date column in sample output: {sample_ticker / 'data.parquet'}")
    print(f"  -> {sep_final}/  ({n_written} ticker dirs)")


# ---------------------------------------------------------------------------
# Step 3: sf1.parquet
# ---------------------------------------------------------------------------

def process_sf1(raw_dir: Path, out_dir: Path) -> None:
    print("[3/3] Processing SF1 (fundamentals)...")
    zip_path = _find_zip(raw_dir, "SHARADAR_SF1")
    print(f"  Reading {zip_path.name} ...")
    df = _read_zip_csv(zip_path)

    for col in ("datekey", "calendardate", "reportperiod", "lastupdated"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "dimension" in df.columns and "ticker" in df.columns:
        df = df.sort_values(["ticker", "datekey", "calendardate"]).reset_index(drop=True)

    out_path = out_dir / "sf1.parquet"
    df.to_parquet(out_path, index=False)
    _spot_check_parquet(out_path, columns=[c for c in ["ticker", "dimension"] if c in df.columns])
    dims = df["dimension"].value_counts().to_dict() if "dimension" in df.columns else {}
    print(f"  -> {out_path}  ({len(df):,} rows, {df['ticker'].nunique():,} tickers)")
    print(f"     dimensions: {dims}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sharadar ZIPs into local parquet bundle")
    parser.add_argument("--raw-dir", required=True, help="Directory containing Sharadar ZIP files")
    parser.add_argument("--out-dir", required=True, help="Output directory for processed parquet files")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise SystemExit(f"raw-dir not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _warn_disk_space(out_dir)

    print(f"raw-dir : {raw_dir}")
    print(f"out-dir : {out_dir}")
    print()

    process_tickers(raw_dir, out_dir)
    print()
    process_sep(raw_dir, out_dir)
    print()
    process_sf1(raw_dir, out_dir)
    print()

    # Summary
    print("=== Output summary ===")
    for item in sorted(out_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / 1024 ** 2
            print(f"  {item.name}: {size_mb:.1f} MB")
        elif item.is_dir():
            n = sum(1 for _ in item.iterdir())
            print(f"  {item.name}/: {n} subdirs")


if __name__ == "__main__":
    main()
