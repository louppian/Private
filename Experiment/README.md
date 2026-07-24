# Experiment — 실험 코드 (npjDM2026 A1 검증 병합)

Private repo 구조:

```
Model/         순수 아키텍처 라이브러리 (BSNet·PAFE·DORGA·SegSTN·common)
Experiment/    학습 엔진(core) + 실험 드라이버(E/A/L) + 실험별 summary
Result/  A/L/E summary 산출 CSV               (✅ git 추적)
checkpoint/    가중치 + raw run(json/npz)      (❌ .gitignore, *.pth 전역 무시)
```

`core.py` = 자립 학습 엔진(split·dataset·train·loss·eval, 3모델 공통). 舊 `dorga_train_2026to2024.py`를
외부 의존 없이 대체. `E/e_common.py`가 `import core as B`로 이를 코어로 쓴다.

---

## 세 축은 동급이 아니다 — E → A → L 계층

- **E** = 실험 (runs 생성층)
- **A** = 그 실험들로 검증하는 **식별가정** (판정층)
- **L** = 수렴 논증의 **증거 사다리** (분석층) — 전부 코드로 계산

영역 순서는 전 코드 **[RT, LT, RB, LB]** 고정. 연도 = `patient_id` 접두어 `24_`/`26_`.

### E — 실험 (draft §5.2)

| 라벨 | 의미 | 코드 |
|---|---|---|
| **E1** | 실데이터 cross (2024↔2026) → δ_obs=(rev−fwd)/2 산출 | `E/e1_cross.py` |
| **E2** | in-domain Δg (raw + 분포정합 matched 통합) → A1 검정 | `E/e2_indomain.py` |
| **E3** | 반합성 양성대조 → 오프셋 β 복원곡선(추정기 무편향) | `E/e3_positive_control.py` |
| **E4** | 반합성 음성대조 → 학습량 비대칭 δ 누출 정량화 | `E/e4_negative_control.py` |
| 공용 | split·dataset·train·bootstrap·δ/γ 분해 | `E/e_common.py` + `core.py` |
| 집계 | fwd/rev × 모델 × ROI bias 표 | `E/summary.py` → `Result/E/` |

E2 는 `raw`(전 환자, 舊 E1·부록 B)와 `matched`(등급분포 정합, 舊 E2·§5.2 본문)를 한 파일에서
돌려 `Δg_raw → Δg_matched`(수축분 제거)까지 낸다. `--raw_only` 로 raw 만도 가능.

### A — 식별가정 (draft §5)

A는 실험이 아니라 **E 산출물로 검증하는 가정**이라 번호별 실험파일이 없다. A1 판정은 E1~E4를 돌려 수행.

| 라벨 | 의미 | 코드 |
|---|---|---|
| **A1** | 오차 방향 대칭성 gᶠ=gʳ | `A/run_all.py`(E3→E1→E2→E4 판정), `A/recompute_bestval.py`(δ best_val 재산출) |
| **A2** | 부분/완전 흡수 | 논문 논증 (E2 결과로 판정) |
| **A3** | 코호트 성분 사전 제거 | 논문 논증 |
| 집계 | δ_obs·Δg·δ_corr 보정표 | `A/summary.py` → `Result/A/` |

### L — 증거 사다리 (draft §4, 전부 코드 계산)

| 라벨 | 계산 | 코드 | 상태 |
|---|---|---|---|
| **L0–L1** 구조·분포 | 환자·영상수, 시퀀스길이, 등급분포 (labels.csv) | `L/l1_structure.py` | ✅ 구현 |
| **L2** 라벨-영상 정합 | ROI **16-특징**(1차4 + GLRLM6 + GLSZM6, 텍스처 직접구현), 인접등급 방향무관 AUC (모델 배제) | `L/l2_label_image.py` | ✅ 구현 |
| **L3** 방향반전 분해 | fwd/rev bias, ROI δ(라벨)/γ(모델) 분해, 환자 부트스트랩 CI (모델 예측) | `L/l3_features.py` + `E/e_common.py:decompose` | ⚠ 학습 의존 |
| **L4** 재현성 | 자기일치 ACC, weighted κ, Wilcoxon (κ 역설) | `L/l4_reproducibility.py` | ✅ 구현(재측정 데이터 대기) |
| 집계 | 위 레벨 표 → CSV | `L/summary.py` → `Result/L/` |

파일명은 파이썬 관례대로 **소문자** 통일(`e*`, `l*`). 폴더는 축 라벨이라 대문자 `A/L/E`.

---

## 실행

```bash
# 서버 검증(병합·재배선 실동작 확인: import + split + dataset + 3모델 forward)
cd <repo 루트> && python temp.py

# E1 cross (checkpoint/E1/runs/dorga/) — 50ep 고정
python Experiment/E/e1_cross.py --seeds 42 1 2

# A1 전체 판정 (E1→E2→E3→E4 → δ 보정)
python Experiment/A/run_all.py

# 집계 → Result/*/
python Experiment/E/summary.py
```

## A1 판정 규칙 (draft §6)

| E1·E2 결과 | A1 | δ 처리 | RB·LT 결론 |
|---|---|---|---|
| Δg ≈ 0 (CI 0 포함) | 성립 | δ 그대로 신뢰 | 라벨 드리프트 확정 |
| Δg ≠ 0, E2 정합 후 소멸 | 조건부 | 수축분 제거 후 사용 | 대체로 유지 |
| Δg ≠ 0, E2 후 잔존 | 위반 | δ_corr = δ_obs + Δg/2 | 잔존 δ 만큼만 주장 |

## 미포팅 / 대기

- **학습 의존(L3·A1)**: `L/l3_features.py`, `L/l3_features_matched.py`, `A/recompute_bestval.py` —
  구버전 base API(`raw_path_from_npz`·`build_2026_cache`·`InhaUHMaskDataset`·`evaluate`)를
  core API로 치환 + E1 cross 산출 필요.
- **L1·L2·L4 는 학습 불필요 — 서버에서 바로 실행**: `l1_structure.py`(labels.csv),
  `l2_label_image.py`(images+masks), `l4_reproducibility.py`(판독자 재측정 CSV `--rr` 필요).
