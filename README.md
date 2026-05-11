# Quant System

NASDAQ small/mid-cap 대상 systematic equity trading 시스템.
Multi-factor + Event-driven (PEAD) 하이브리드 구조를 목표로 한다.

## 설치

### pip
```bash
pip install -r requirements.txt
```

### Poetry
```bash
poetry install
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

### Config 확인
```bash
python -c "from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.data.backend)"
```

## 데이터 Backend

기본값은 `local`이다. 팀 공유 Sharadar 전처리 데이터나 Sharadar API를 쓸 때는 repo root에 `.env.local`을 만들고 개인 환경값만 넣는다. `.env.local`은 git에 올리지 않는다.

### 로컬 전처리 데이터
```env
QUANT_DATA_BACKEND=local
NASDAQ_DATA_DIR=C:\path\to\nasdaq_data\processed
```

`NASDAQ_DATA_DIR`에는 로컬 preprocess 폴더 경로를 넣는다.

### Sharadar API
```env
QUANT_DATA_BACKEND=sharadar
NASDAQ_DATA_LINK_API_KEY=your_api_key
```

`QUANDL_API_KEY`도 대체 키 이름으로 지원한다.

## 디렉토리 구조

| 경로 | 설명 |
|---|---|
| `core/` | 공통 설정, universe 구성 |
| `data_layer/` | local parquet, Polygon, Sharadar 데이터 backend |
| `backtesting/` | 공통 백테스트 엔진, 거래비용, trend filter |
| `strategies/` | 전략별 구현과 registry |
| `strategies/multifactor/` | 4/5-factor multi-factor 전략 |
| `trading/` | 실행 엔진, risk 엔진 |
| `research/` | 분석, 최적화, robustness, walk-forward |
| `ui/` | Streamlit tab별 UI |
| `tests/` | pytest 단위 테스트 |

기존 루트 파일(`data.py`, `backtest.py`, `factors.py`, `portfolio.py` 등)은 호환용 wrapper로 유지한다. 새 코드는 가능하면 패키지 경로를 직접 import한다.

## 전략 추가 규칙

새 전략은 `strategies/{strategy_name}/` 폴더로 만든다.

1. 전략별 로직은 해당 전략 폴더 안에서만 수정한다.
2. 공통 백테스트 로직은 `backtesting/`에서 수정한다.
3. UI나 백테스트에서 선택하려면 `strategies/registry.py`에 등록한다.
4. 새 패키지를 쓰면 `requirements.txt`에 추가한다.
5. 로컬 경로나 개인 PC 의존성은 `requirements.txt`에 넣지 않는다.

## 현재 상태

- 폴더 구조 정리 완료: `core`, `data_layer`, `backtesting`, `strategies`, `trading`, `research`, `ui`
- 기존 루트 파일은 호환용 wrapper로 유지
- multi-factor 전략은 `strategies/multifactor`로 분리
- 전략 registry 추가
- `requirements.txt` 추가
- SPY/QQQ benchmark 비교 지원
- local parquet backend 및 Sharadar backend 지원
- 파이프라인 테스트 통과: `pytest`
