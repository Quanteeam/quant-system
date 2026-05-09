"""yfinance 데이터 로딩 + 캐싱.

캐시: ~/.cache/quant-system/prices_{hash}.parquet
배치: 50종목씩, 실패 시 3회 retry
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".cache" / "quant-system"
BATCH_SIZE = 50
MAX_RETRIES = 3


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    payload = json.dumps({"tickers": sorted(tickers), "start": start, "end": end})
    h = hashlib.md5(payload.encode()).hexdigest()
    return CACHE_DIR / f"prices_{h}.parquet"


def _download_batch(batch: list[str], start: str, end: str) -> pd.DataFrame:
    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(batch, start=start, end=end, auto_adjust=False, progress=False)
            if raw.empty:
                raise ValueError("Empty response from yfinance")
            return raw
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise


def _normalize_columns(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance (field, ticker) MultiIndex → (ticker, field), lowercase."""
    df = raw.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker: flat columns like ['Open', 'Close', ...]
        ticker = tickers[0]
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, c.lower().replace(" ", "_")) for c in df.columns],
            names=["ticker", "field"],
        )
    else:
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, field.lower().replace(" ", "_")) for field, ticker in df.columns],
            names=["ticker", "field"],
        )
    return df.sort_index(axis=1)


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """yfinance daily OHLCV.

    Returns:
        MultiIndex columns (ticker, field) DataFrame.
        field = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    Cache:
        ~/.cache/quant-system/prices_{hash}.parquet
    """
    path = _cache_key(tickers, start, end)
    if path.exists():
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    frames = [_normalize_columns(_download_batch(b, start, end), b) for b in batches]
    df = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Fundamental data (Phase 2)
# ---------------------------------------------------------------------------
_FUND_CACHE_TTL = 86400  # 24시간


def load_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """yfinance ticker.info에서 fundamental 데이터 로딩.

    Returns:
        DataFrame indexed by ticker. Columns:
        market_cap, pe_ratio, pb_ratio, ev_ebitda, fcf_yield,
        roe, gross_margin, debt_equity, sector

    ⚠ 현재 시점 스냅샷 → look-ahead bias. Phase 6에서 PIT 데이터로 교체.
    Cache: ~/.cache/quant-system/fund_{hash}.parquet (24h TTL)
    """
    h = hashlib.md5(json.dumps(sorted(tickers)).encode()).hexdigest()
    path = CACHE_DIR / f"fund_{h}.parquet"

    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap") or 0
            rows[ticker] = {
                "market_cap": mc,
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "fcf_yield": (info.get("freeCashflow") or 0) / mc if mc > 0 else None,
                "roe": info.get("returnOnEquity"),
                "gross_margin": info.get("grossMargins"),
                "debt_equity": info.get("debtToEquity"),
                "sector": info.get("sector", "Unknown"),
            }
        except Exception:
            continue

    df = pd.DataFrame(rows).T
    numeric_cols = [c for c in df.columns if c != "sector"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# PIT (Point-in-Time) Fundamentals
# ---------------------------------------------------------------------------

def _merge_financials(quarterly: pd.DataFrame | None,
                       annual: pd.DataFrame | None) -> pd.DataFrame:
    """분기 + 연간 재무제표 병합. 분기 우선, 연간으로 과거 보충."""
    frames = []
    if quarterly is not None and not quarterly.empty:
        frames.append(quarterly)
    if annual is not None and not annual.empty:
        # 분기 데이터에 이미 있는 날짜는 제외
        existing = set(quarterly.columns) if quarterly is not None and not quarterly.empty else set()
        annual_new = annual[[c for c in annual.columns if c not in existing]]
        if not annual_new.empty:
            frames.append(annual_new)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index(axis=1)


def load_quarterly_cache(tickers: list[str]) -> dict:
    """분기+연간 재무제표 다운로드 + 캐시 (pickle, 24h TTL).

    Returns: {ticker: {income, balance, cashflow, sector}}
    연간 데이터로 과거(~2021)까지 커버, 분기 데이터로 최근 정밀도 확보.
    """
    h = hashlib.md5(json.dumps(sorted(tickers)).encode()).hexdigest()
    path = CACHE_DIR / f"quarterly_v2_{h}.pkl"

    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        with open(path, "rb") as f:
            return pickle.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            qi = t.quarterly_income_stmt
            ai = t.income_stmt
            qb = t.quarterly_balance_sheet
            ab = t.balance_sheet
            qc = t.quarterly_cashflow
            ac = t.cashflow
            sector = t.info.get("sector", "Unknown")

            inc = _merge_financials(qi, ai)
            bs = _merge_financials(qb, ab)
            cf = _merge_financials(qc, ac)

            if not inc.empty:
                data[ticker] = {"income": inc, "balance": bs, "cashflow": cf,
                                "sector": sector}
        except Exception:
            continue

    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


def _safe_get(df: pd.DataFrame, keys: list[str], col) -> float | None:
    """재무제표에서 여러 가능한 행 이름으로 값 추출."""
    if df is None or df.empty or col not in df.columns:
        return None
    for key in keys:
        if key in df.index:
            val = df.loc[key, col]
            if pd.notna(val):
                return float(val)
    return None


def get_pit_fundamentals(
    quarterly_cache: dict,
    close_at_date: pd.Series,
    as_of_date: pd.Timestamp,
    lag_days: int = 90,
) -> pd.DataFrame:
    """as_of_date 기준 PIT 펀더멘탈 재구성.

    lag_days: 분기 종료 후 데이터 공시까지 lag (보수적 90일).
    close_at_date: ticker → 리밸런스일 종가.
    """
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=lag_days)

    rows: dict[str, dict] = {}
    for ticker, qd in quarterly_cache.items():
        inc, bs, cf = qd["income"], qd["balance"], qd["cashflow"]

        avail = [c for c in inc.columns if pd.Timestamp(c) <= cutoff]
        if not avail:
            continue
        latest = max(avail, key=lambda x: pd.Timestamp(x))

        price = close_at_date.get(ticker)
        if price is None or pd.isna(price) or price <= 0:
            continue

        # 재무 항목 추출
        revenue = _safe_get(inc, ["Total Revenue", "Revenue"], latest)
        gross_profit = _safe_get(inc, ["Gross Profit"], latest)
        net_income = _safe_get(inc, ["Net Income", "Net Income Common Stockholders"], latest)
        equity = _safe_get(bs, ["Stockholders Equity", "Total Stockholder Equity",
                                "Stockholders' Equity", "Common Stock Equity"], latest)
        total_debt = _safe_get(bs, ["Total Debt", "Long Term Debt"], latest)
        fcf_val = _safe_get(cf, ["Free Cash Flow"], latest)
        shares = _safe_get(bs, ["Ordinary Shares Number", "Share Issued"], latest)
        ebitda = _safe_get(inc, ["EBITDA", "Normalized EBITDA"], latest)

        # Trailing 4Q net income (PE 계산용)
        avail_sorted = sorted([c for c in inc.columns if pd.Timestamp(c) <= cutoff],
                              key=lambda x: pd.Timestamp(x), reverse=True)[:4]
        trailing_ni = 0.0
        for q in avail_sorted:
            ni = _safe_get(inc, ["Net Income", "Net Income Common Stockholders"], q)
            if ni is not None:
                trailing_ni += ni

        # 비율 계산
        mc = price * shares if shares and shares > 0 else None

        pe = (mc / trailing_ni) if mc and trailing_ni and trailing_ni > 0 else None
        pb = (mc / equity) if mc and equity and equity > 0 else None
        roe = (net_income * 4 / equity) if net_income and equity and equity > 0 else None
        gm = (gross_profit / revenue) if gross_profit and revenue and revenue > 0 else None
        de = (total_debt / equity) if total_debt is not None and equity and equity > 0 else None
        ev_ebitda_val = None
        if mc and ebitda and ebitda > 0:
            ev = mc + (total_debt or 0)
            ev_ebitda_val = ev / (ebitda * 4)
        fcf_y = (fcf_val * 4 / mc) if fcf_val and mc and mc > 0 else None

        rows[ticker] = {
            "market_cap": mc, "pe_ratio": pe, "pb_ratio": pb,
            "ev_ebitda": ev_ebitda_val, "fcf_yield": fcf_y,
            "roe": roe, "gross_margin": gm, "debt_equity": de,
            "sector": qd["sector"],
        }

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    numeric_cols = [c for c in df.columns if c != "sector"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Earnings data (Phase 4 — PEAD)
# ---------------------------------------------------------------------------


def load_earnings(tickers: list[str]) -> pd.DataFrame:
    """yfinance에서 실적 발표 이력 로딩.

    Returns:
        DataFrame: ticker, date, actual_eps, estimate_eps.
        date는 tz-naive, normalized (시간 제거).

    ⚠ yfinance는 최근 4~8분기만 제공. Phase 6에서 Sharadar PIT로 교체.
    Cache: ~/.cache/quant-system/earnings_{hash}.parquet (24h TTL)
    """
    h = hashlib.md5(json.dumps(sorted(tickers)).encode()).hexdigest()
    path = CACHE_DIR / f"earnings_{h}.parquet"

    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ticker in tickers:
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is None or ed.empty:
                continue
            for dt, row in ed.iterrows():
                actual = row.get("Reported EPS")
                estimate = row.get("EPS Estimate")
                if pd.isna(actual) or pd.isna(estimate):
                    continue
                rows.append({
                    "ticker": ticker,
                    "date": pd.Timestamp(dt).tz_localize(None).normalize(),
                    "actual_eps": float(actual),
                    "estimate_eps": float(estimate),
                })
        except Exception:
            continue

    df = (pd.DataFrame(rows) if rows
          else pd.DataFrame(columns=["ticker", "date", "actual_eps", "estimate_eps"]))
    df.to_parquet(path)
    return df
