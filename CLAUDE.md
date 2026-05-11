# Quant Trading System

NASDAQ small/mid-cap 대상 systematic equity trading 시스템.
Multi-factor + Event-driven (PEAD) 하이브리드 구조.

---

## 시스템 개요 (선택지 B: 40/60 hybrid)

자본을 두 sleeve로 분리:

- **Multi-factor sleeve (40%)** — 5팩터 동등가중, 20종목, monthly rebalance
- **Event-driven sleeve (60%)** — PEAD 트리거 + multi-factor loose filter, max 40종목

목표: S&P 500 대비 연 +2~5% 알파 (net of costs), max drawdown < -20%, Sharpe > 0.6.

---

## 디렉토리 구조

```
quant-system/
├── CLAUDE.md           # 이 파일
├── WORKFLOW.md         # 사용자용 진행 가이드
├── README.md           # Phase 1에 생성
├── pyproject.toml
├── config.py           # 모든 파라미터 (dataclass)
├── data.py             # 데이터 backend 호환 wrapper
├── factors.py          # 5팩터 + SUE 계산
├── portfolio.py        # 60/40 sleeve construction
├── backtest.py         # 백테스트 엔진 + metrics
├── risk.py             # Risk engine (Phase 5)
├── app.py              # Streamlit UI
├── main.py             # CLI 진입점
└── tests/              # pytest
```

---

## 코드 스타일

- Python 3.11+
- 타입 힌트 필수 (`from __future__ import annotations`)
- `dataclass` 또는 `pydantic`으로 데이터 구조 명시
- 함수 docstring 필수
- 의존성 관리: `pyproject.toml` (poetry 선호)
- 포매터: black, isort
- 린터: ruff

---

## 모델 사용 정책 (Opus vs Sonnet)

비용 효율을 위해 작업 성격에 따라 모델 분리. **기준: 이 코드가 틀리면 돈을 잃는가?**

### Opus 사용 영역 (정확성 critical)

- `factors.py` 본 구현 — look-ahead bias, sector neutral z-score, SUE 계산
- `backtest.py` 엔진 — fill timing, walk-forward, Sharpe/drawdown 계산식
- `risk.py` 전체 — pre-trade check, drawdown halt, kill switch
- `portfolio.py` 의 `combine_sleeves()`, position sizing, vol target
- `data.py` 의 corporate action 처리 (split, dividend, delisting)
- PEAD signal 로직 — earnings date alignment, D+1 진입, 청산 우선순위
- Live execution 코드 (Phase 7) — broker state 동기화

### Sonnet 사용 영역 (검증 쉬움 / 표준 패턴)

- `app.py` Streamlit UI 전체
- `data_layer/` backend 연결, 로컬 데이터 로딩, 캐싱/에러 처리
- `main.py` CLI 진입점
- `tests/` 단위 테스트 작성 (테스트 케이스 설계는 Opus)
- `pyproject.toml`, `README.md` 보일러플레이트
- `factors.py` NotImplementedError stub
- Plotly chart 디테일

### 디버깅 모델 선택

- Syntax/import 에러 → Sonnet
- 백테스트 결과 이상함, 알파 의심 → Opus
- UI 깨짐 → Sonnet
- Live trading 동작 이상 → Opus

### Phase별 모델 비중

| Phase | Opus | Sonnet |
|---|---|---|
| 1 (셋업, UI skeleton) | 10% | 90% |
| 2 (5팩터 계산) | 70% | 30% |
| 3 (multi-factor 백테스트) | 60% | 40% |
| 4 (PEAD + event sleeve) | 80% | 20% |
| 5 (risk engine) | 90% | 10% |
| 6 (Polygon 마이그레이션) | 50% | 50% |
| 7 (IBKR live) | 80% | 20% |

---

## 의존성

핵심: `pandas`, `numpy`, `pyarrow`, `streamlit`, `plotly`
데이터: 로컬 Sharadar parquet bundle 또는 Nasdaq Data Link Sharadar API
미래: `ib_async` (Phase 7+, `ib_insync`는 폐기)

---

## 데이터 소스

| 용도 | 소스 | 설정 |
|---|---|---|
| 기본 개발/백테스트 | 로컬 Sharadar parquet bundle | `QUANT_DATA_BACKEND=local`, `NASDAQ_DATA_DIR=...` |
| API 직접 조회 | Nasdaq Data Link Sharadar | `QUANT_DATA_BACKEND=sharadar`, `NASDAQ_DATA_LINK_API_KEY=...` |
| PEAD consensus 확장 | Estimize/Zacks 등 | 별도 backend 추가 |

`yfinance`는 사용하지 않는다. SPY/QQQ benchmark도 로컬 데이터 bundle에 포함되어 있어야 한다.

---

## Universe

- Index: Russell 2000 + Russell Midcap 교집합 (약 1500종목)
- Min price $5, min 20-day ADV $5M, OTC/ADR 제외
- Phase 1: S&P 500 종목 중 시총 하위 50개 proxy

---

## Multi-factor sleeve (40%)

| Factor | 정의 | 가중치 |
|---|---|---|
| Size | -log(market_cap) | 0.20 |
| Value | composite z(PER, PBR, EV/EBIT, FCF yield) | 0.20 |
| Momentum | 12-1 month return | 0.20 |
| Quality | composite z(ROE, gross margin, low leverage) | 0.20 |
| Low Vol | -volatility(60d) | 0.20 |

- 5팩터 z-score 평균 → 종목 score
- Sector neutral ranking, top 20 equal weight, monthly rebalance

---

## Event-driven sleeve (60%)

### Trigger: PEAD

- SUE = (Actual EPS - Consensus EPS) / std(historical surprises)
- **SUE > +1.5** 후보

### Loose filter

- Quality z > sector median
- Value z > sector median
- 직전 30일 negative analyst revision 없음
- 직전 10일 다른 발표 없음

### Entry / Exit

- Entry: 발표 D+1, IBKR Adaptive algo
- Position: 1.5% per stock, max 40종목
- Exit (whichever first): D+45 / 다음 실적 D-3 / -10% stop loss

활성 진입 없을 때 cash 또는 SHY.

---

## Risk Engine (Phase 5)

### Pre-trade

| 한도 | 값 |
|---|---|
| 단일 종목 | 3% |
| Sector | 30% |
| ADV % | 5% (초과 시 분할) |
| Daily VaR 95% | 2% |

### Real-time

| 트리거 | 액션 |
|---|---|
| 일일 손실 -3% | 신규 주문 정지 24h |
| Drawdown -10% | 알람 |
| Drawdown -15% | Event sleeve 50% 축소 |
| Drawdown -20% | 전 시스템 halt |

### Sanity

- 전일 대비 ±30% 변동 → reject
- Stale 데이터 N분+ → 정지
- NaN/zero/inf 제거

### Kill switch

`python main.py kill` → 전 포지션 청산 + 신규 주문 차단.

---

## 백테스트 검증 (4 baseline 필수)

| Baseline | 의미 |
|---|---|
| SPY buy & hold | 시장 |
| Multi-factor 100% | factor sleeve 단독 |
| PEAD 100% | event sleeve 단독 |
| 40/60 hybrid | 본 시스템 |

**통과 기준**: hybrid Sharpe > (multi-factor only, PEAD only) 둘 다,
max DD < PEAD only.

### Walk-forward

- Train 5년 / Test 1년, 2014~2024 6 walks
- In-sample vs out-of-sample Sharpe 차이 < 30%

---

## UI (Streamlit)

### Sidebar

- Date range, Multi-factor allocation slider, SUE threshold slider
- 5팩터 가중치 슬라이더, PEAD holding period, stop loss
- "Run backtest" 버튼

### Main

1. Top metric row — CAGR, Sharpe, Max DD, Calmar (4 baseline)
2. Equity curve (4 baseline)
3. Drawdown chart (4 baseline)
4. Monthly returns heatmap
5. Position breakdown
6. Trade log

`@st.cache_data` 캐싱 필수.

---

## 절대 금지

1. **시장가 주문**: Limit 또는 IBKR Adaptive algo만.
2. **Look-ahead bias**: t 시점 신호 → t+1 진입.
3. **Survivorship bias**: 현재 universe로 과거 백테스트 금지.
4. **Future fundamental**: 발표일 publish 이후만.
5. **Single-test optimization**: Walk-forward 필수.
6. **Real broker API in tests**: Mock 사용.

---

## Phase 진행

| Phase | 내용 | 검증 |
|---|---|---|
| 1 | 셋업 + UI skeleton + mock 백테스트 | `streamlit run app.py` 동작 |
| 2 | 5팩터 계산 (Sharadar PIT/local) | factor z-score 분포 정상 |
| 3 | Multi-factor sleeve 백테스트 | Sharpe > SPY |
| 4 | PEAD signal + event sleeve | 단위 테스트 통과 |
| 5 | 60/40 통합 + Risk Engine | 4 baseline 비교 |
| 6 | Polygon + Sharadar 마이그레이션 | PIT 재검증 |
| 7 | IBKR paper trading | 8주 paper → live Sharpe ≥ 70% backtest |

각 Phase 미달 시 다음 가지 마라.

---

## 변경 원칙

- 한 번에 한 모듈씩
- 변경 사유를 commit message에 명시
- 함수 시그니처 변경 시 호출자 모두 수정
- 각 파일 200줄 이내 (넘으면 분리 검토)
