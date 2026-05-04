# Quant System

NASDAQ small/mid-cap 대상 systematic equity trading 시스템.
Multi-factor + Event-driven (PEAD) 하이브리드 구조 (40/60 split).

## 설치

### Poetry (권장)
```bash
poetry install
```

### pip
```bash
pip install pandas numpy yfinance streamlit plotly python-dateutil pyarrow
pip install pytest ruff black  # dev
```

## 실행

### Streamlit UI
```bash
streamlit run app.py
# 또는
python main.py ui
```

### CLI 백테스트
```bash
python main.py backtest
```

### 테스트
```bash
pytest
```

### config 확인
```bash
python -c "from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.pead.sue_threshold)"
```

## 디렉토리

| 파일 | 설명 |
|---|---|
| `config.py` | 전체 시스템 파라미터 (수정 금지) |
| `data.py` | yfinance 데이터 로딩 + 캐싱 |
| `factors.py` | 팩터 계산 (Phase 1: momentum) |
| `portfolio.py` | 포트폴리오 구성 (multi-factor / event sleeve) |
| `backtest.py` | 백테스트 엔진 + metrics |
| `app.py` | Streamlit UI |
| `main.py` | CLI 진입점 |
| `tests/` | pytest 단위 테스트 |

## Phase 진행

- [x] Phase 1 — 셋업 + UI skeleton + momentum 백테스트
- [ ] Phase 2 — 5팩터 실제 계산 (Size, Value, Quality, Low Vol)
- [ ] Phase 3 — Multi-factor sleeve 백테스트 검증
- [ ] Phase 4 — PEAD signal + event sleeve
- [ ] Phase 5 — 60/40 통합 + Risk Engine
- [ ] Phase 6 — Polygon + Sharadar 마이그레이션
- [ ] Phase 7 — IBKR paper trading

## 다음: Phase 2 — 5팩터 실제 계산

`factors.py`에서 `compute_size`, `compute_value`, `compute_quality`, `compute_lowvol` 구현.
