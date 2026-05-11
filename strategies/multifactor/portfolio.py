"""Portfolio construction for the active multi-factor workflow."""
from __future__ import annotations

import pandas as pd


def build_multifactor_portfolio(
    scores: pd.Series,
    top_n: int = 20,
) -> pd.Series:
    """Select top-N names and assign equal weights."""
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=float)

    top = valid.nlargest(min(top_n, len(valid)))
    weight = 1.0 / len(top)
    return pd.Series(weight, index=top.index)
