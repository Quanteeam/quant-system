# Phase 1 Prompt — 첫 명령

> **권장 모델: Sonnet** (셋업 + UI skeleton, 보일러플레이트가 90%)
> 
> 사용법: `claude --model claude-sonnet-4-6` 시작 후 이 파일 내용 전체 복붙.
> 또는 세션 중 `/model sonnet` 으로 전환.

---

이 프로젝트의 마스터 명세는 `CLAUDE.md`에 있다. 작업 시작 전 먼저 읽어라.

## Phase 1 목표

End-to-end skeleton 만들기. **알고리즘은 mock이어도 되니, 실행 시 Streamlit UI에서 백테스트 차트가 브라우저에 보이는 것**이 검증 기준.

코드 품질보다 **흐름이 통하는 것**이 우선. Phase 2부터 실제 알고리즘 채움.

---

## 작업

### 1. `pyproject.toml`
- Python 3.11+
- 의존성: `pandas`, `numpy`, `yfinance`, `streamlit`, `plotly`, `python-dateutil`, `pyarrow`
- Dev: `pytest`, `ruff`, `black`
- Poetry 또는 pip-tools.

### 2. `config.py`
이미 존재. 손대지 마라.

### 3. `data.py`
```python
def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """yfinance daily OHLCV.

    Returns: MultiIndex columns (ticker, field) DataFrame.
             field = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    Cache: ~/.cache/quant-system/prices_{hash}.parquet
    """
```
- 50종목씩 batch, 실패 시 3회 retry
- 캐시 hit 시 disk 읽기
- 캐시 키: `(tickers, start, end)` 해시

### 4. `factors.py`
Momentum만 실제 구현. 나머지는 stub:

```python
def compute_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """12-1 momentum: 12개월 누적, 직전 1개월 제외."""

def compute_size(prices: pd.DataFrame, market_caps: pd.Series) -> pd.Series:
    raise NotImplementedError("Phase 2")

def compute_value(...) -> pd.Series:
    raise NotImplementedError("Phase 2")

def compute_quality(...) -> pd.Series:
    raise NotImplementedError("Phase 2")

def compute_lowvol(prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
    raise NotImplementedError("Phase 2")

def compute_sue(...) -> pd.Series:
    raise NotImplementedError("Phase 3")
```

### 5. `portfolio.py`
```python
def build_multifactor_portfolio(scores: pd.Series, top_n: int = 20) -> pd.Series:
    """Top N equal weight. Returns: ticker -> weight."""

def build_event_portfolio(...) -> pd.Series:
    raise NotImplementedError("Phase 4")

def combine_sleeves(mf_weights: pd.Series, event_weights: pd.Series,
                    mf_alloc: float = 0.4, event_alloc: float = 0.6) -> pd.Series:
    """두 sleeve를 alloc 비율로 통합. 동일 종목 weight 합산."""
```

### 6. `backtest.py`
```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series
    drawdown: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    benchmark_curve: pd.Series

class BacktestEngine:
    def __init__(self, prices: pd.DataFrame, initial_capital: float = 100_000):
        ...

    def run(self, weights_history: pd.DataFrame) -> BacktestResult:
        """weights_history: index=date, columns=tickers, values=target weight.
        Phase 1: daily rebalance, commission/slippage 무시."""
```

### 7. `app.py` (Streamlit UI)
```
[Sidebar]
- Date range picker (기본: 2020-01-01 ~ 2024-12-31)
- Top N stocks slider (10~50, 기본 20)
- Momentum lookback (60~252, 기본 252)
- "Run backtest" 버튼

[Main]
- 4 metric cards: CAGR, Sharpe, Max DD, Total Return
- Plotly equity curve: 본 시스템 + SPY
- Plotly drawdown chart (area, 빨강)
- 종목별 weight 테이블
```

데이터: S&P 500 종목 중 시총 하위 50개 하드코딩 (`UNIVERSE_TICKERS = [...]`).
yfinance로 시총 가져와서 정렬 후 fix.

`@st.cache_data` 캐싱 필수.

### 8. `main.py`
```python
"""CLI 진입점."""
# python main.py backtest    → 콘솔 백테스트
# python main.py ui          → streamlit run app.py
# python main.py kill        → NotImplementedError (Phase 5+)
```

### 9. `README.md`
- 설치, 실행 방법
- 디렉토리 설명
- Phase 진행 체크박스
- 다음: "Phase 2 — 5팩터 실제 계산"

### 10. `tests/`
```
tests/
├── test_data.py        # load_prices 모킹 (yfinance 호출 X)
├── test_factors.py     # compute_momentum 단순 케이스
└── test_portfolio.py   # build_multifactor_portfolio
```

각 3~5 케이스만.

---

## 검증 (Phase 1 종료 조건)

다음이 **모두** 통과:

1. `poetry install` 또는 `pip install -e .` 성공
2. `python -c "from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.pead.sue_threshold)"` → `1.5`
3. `pytest` 통과
4. `streamlit run app.py` → 브라우저 열림 → 차트 표시
5. Sidebar 파라미터 변경 → "Run backtest" → 차트 갱신

---

## 제약

- 각 파일 **200줄 이내**
- yfinance survivorship bias 알고 있다 (Phase 6 교체)
- 시장가 주문 코드 절대 금지 (Phase 7에서 Adaptive algo)
- Look-ahead 주의: 시점 t 신호는 t+1 진입에만
- 백테스트 결과 mock OK, 흐름은 정확해야 (weights → returns → equity)

---

## 작업 순서

1. `pyproject.toml`
2. 빈 파일들 생성
3. `data.py`
4. `factors.py` momentum
5. `portfolio.py` multifactor
6. `backtest.py` 단순
7. `app.py` UI
8. `main.py` CLI
9. `tests/`
10. `README.md`

각 단계 후 `python -c "import [module]"` import 에러 확인.

---

## 끝나면 보고

```
Phase 1 완료. 검증:
  poetry install   (또는 pip install -e .)
  pytest
  streamlit run app.py
```
