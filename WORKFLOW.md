# Workflow Guide

이 문서는 너(사용자)가 따라가는 가이드. Claude Code가 보는 건 `CLAUDE.md`.

---

## 0. 초기 셋업 (1회만)

```bash
mkdir quant-system && cd quant-system

# 다운로드한 파일 3개를 이 디렉토리로 이동
# - CLAUDE.md
# - config.py
# - WORKFLOW.md (이 파일)

# Git 초기화 (Phase 단위 commit 권장)
git init
git add .
git commit -m "Initial: project spec and config"
```

현재 코드는 Phase 1 프롬프트 기반 skeleton 단계를 지났으므로 `README.md`와 `CLAUDE.md`를 기준으로 진행한다.

---

## 1. Claude Code 모델 정책

비용 차이 5~7배. 작업 종류로 분리.

### Opus 사용 (정확성 critical)
- Factor 계산 (look-ahead 위험)
- 백테스트 엔진 (fill timing, Sharpe 계산)
- Risk engine (자본 보호 직결)
- Live execution (broker 동기화)

### Sonnet 사용 (검증 쉬움 / 표준)
- UI (Streamlit, plotly)
- 데이터 로딩 (local Sharadar parquet, 캐싱)
- CLI, README, pyproject.toml
- 단위 테스트 작성
- Stub 함수

### 시작 시 모델 지정
```bash
claude --model claude-sonnet-4-6   # 기본 (Phase 1)
claude --model claude-opus-4-7     # critical 작업
```

### 세션 중 전환
```
/model sonnet
/model opus
```

### Phase별 시작 모델

| Phase | 시작 모델 |
|---|---|
| 1 (셋업, UI) | **Sonnet** |
| 2 (5팩터) | **Opus** |
| 3 (백테스트) | **Opus**, UI는 Sonnet |
| 4 (PEAD) | **Opus** |
| 5 (Risk) | **Opus** |
| 6 (Polygon) | **Opus**, API 클라이언트는 Sonnet |
| 7 (Live) | **Opus** |

---

## 2. Phase 진행

### Phase 1: 셋업 + UI Skeleton (지금)

```bash
cd quant-system
claude --model claude-sonnet-4-6
# README.md와 CLAUDE.md 기준으로 현재 작업 범위만 요청
```

작업 끝나면 검증:
```bash
poetry install   # or pip install -e .
pytest
streamlit run app.py
```

브라우저에 차트 떠야 통과. 통과 후 commit:
```bash
git add . && git commit -m "Phase 1 complete: UI skeleton + mock backtest"
```

### Phase 2~7

각 Phase 끝나고 결과 보고하면 다음 prompt 작성해줌. 흐름:

1. 직전 Phase 검증 결과 알려줘 (pass/fail, 어디 막혔는지)
2. 다음 Phase prompt 받기
3. Claude Code에서 권장 모델로 시작
4. 작업 후 검증
5. Git commit
6. 반복

---

## 3. 디버깅 워크플로우

### 에러 발생 시 우선순위

| 증상 | 사용 모델 | 액션 |
|---|---|---|
| `ImportError`, `SyntaxError` | Sonnet | 그냥 Claude Code에 에러 메시지 |
| `pytest` 실패 | Sonnet 먼저, 안 풀리면 Opus | 어느 테스트 실패인지 명시 |
| Streamlit 화면 깨짐 | Sonnet | 스크린샷 + 에러 |
| 백테스트 결과 비현실적 (Sharpe 3.0 등) | **Opus** | 의심: look-ahead, lookback bias |
| Walk-forward에서 IS/OS 차이 큼 | **Opus** | overfitting 검증 |
| Live trading 동작 이상 | **Opus** 무조건 | 절대 자체 수정 X |

### 막히면 보고할 것

```
Phase: N
모델: Sonnet/Opus
작업: [무엇을 하다가]
에러:
  [전체 에러 메시지 또는 증상]
시도한 것:
  [무엇을 시도해봤는지]
```

이 형식으로 알려주면 다음 단계 정확히 안내.

---

## 4. 비용 절감 팁

1. **Phase 1은 무조건 Sonnet** — 토큰 90% 절감, 품질 동일
2. **`CLAUDE.md`가 컨텍스트 역할** — Claude Code가 매번 재구성 안 함, 토큰 ↓
3. **한 Phase 끝나면 새 세션 시작** — 컨텍스트 짧게 유지
4. **Git commit 자주** — 롤백 가능, 실수 비용 ↓
5. **Opus에 큰 작업 금지** — 한 모듈씩, 컨텍스트 폭발 방지
6. **테스트는 Sonnet이 작성, Opus가 review** — 작성 비용 ↓
7. **mock/로컬 Sharadar 데이터로 검증** — 개인 경로와 API key는 `.env.local`에만 보관

### Phase별 Anthropic 토큰 예상 비용

> 정확한 가격은 https://docs.claude.com 참고

대략적 비중:
- Phase 1: 매우 적음 (Sonnet 90%)
- Phase 2~3: 중간 (Opus 60~70%)
- Phase 4~5: 많음 (Opus 80~90%, 큰 모듈)
- Phase 6: 중간
- Phase 7: 많음 (Opus + paper trading 디버깅)

**Phase 5까지 가면서 너의 자본이 정당화 안 되면 멈춰라.** 데이터 비용 + Opus 비용 + 본인 시간 모두 합쳐서.

---

## 5. 절대 하지 말아야 할 것

1. **검증 안 하고 다음 Phase로** — 백테스트 결과 안 보고 진행하면 Phase 5에서 디버깅 지옥
2. **Opus로 UI 디테일 수정** — 토큰 낭비
3. **Sonnet으로 risk engine** — 자본 손실 위험
4. **Live 자본 투입 전 paper trading 8주 미만** — paper 결과 확인 없이 실거래 금지
5. **CLAUDE.md 무시한 작업 요청** — Claude Code가 따로 가버림

---

## 6. 자본 규모별 가이드

자본은 데이터 비용 + 거래 효율 결정.

| 자본 | 권장 |
|---|---|
| < $20k | 이 시스템 무의미. SPY ETF 또는 robo-advisor가 합리적. |
| $20k~50k | Phase 1~3까지 학습용. 라이브 가지 마라. |
| $50k~100k | Phase 5까지. 라이브는 paper trading만. |
| $100k~500k | Phase 7 라이브 가능. 단 자본 30%만 시작. |
| $500k+ | 본격 구현 + risk engine 풀 가동 의미 있음. |

---

## 7. 시간 투자 예상

각 Phase 본인이 직접 검증/디버그 하는 시간 (Claude Code 작업 시간 제외):

| Phase | 사용자 시간 |
|---|---|
| 1 | 1~2일 |
| 2 | 2~3일 (factor 검증) |
| 3 | 3~5일 (백테스트 결과 해석) |
| 4 | 5~7일 (PEAD 로직 디버그) |
| 5 | 5~7일 (risk engine 단위 테스트) |
| 6 | 1~2주 (데이터 마이그레이션) |
| 7 | 8주+ (paper trading) |

총 3~4개월 학생 페이스. 본업/공부 균형 잡고 가라.

---

## 다음 액션

```bash
cd quant-system
claude --model claude-sonnet-4-6
# README.md와 CLAUDE.md 기준으로 현재 작업 범위만 요청
```

Phase 1 끝나면 결과 보고 → Phase 2 prompt 작성해주겠다.
