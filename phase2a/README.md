# Phase 2A 유틸리티

`D:\npjDM2026\plan_phase2a.md` 의 정보 비노출 재판독 세트를 구성하고, 사전 점검과
검정력 시뮬레이션을 수행한다.

세 스크립트 모두 **CLI 인자가 없다.** 파일 상단 상수를 직접 고쳐서 쓴다
(`Experiment/L/l2_label_image.py` 와 같은 방식). 산출은 전부 `Result/P2A/` 에
`p2a_*` 이름으로 떨어지고, 같은 내용을 표로 출력한다.

경로는 `core.py` 와 같은 형식으로 상단에 있다. 서버에서는 그대로 돌아간다.

```python
BASE   = "/shared/home/mai/JeongGeon/Private"
MERGED = Path(f"{BASE}/CXR/Merged")                    # image_path 기준 폴더
IMG_DIR = Path(f"{BASE}/CXR/Merged/images_normalize")  # <uid>.png
LABELS_CSV = f"{BASE}/CXR/Merged/labels.csv"
```

## 실행 순서

```bash
python phase2a/precheck_grade_composition.py
python phase2a/build_minisequence.py
python phase2a/power_simulation.py
```

## 1. §3.6 사전 점검

`precheck_grade_composition.py` — 재판독 **전에** 두 연도의 RB 기존 등급 2·3·4
영상 특징 분포를 비교한다. 입력은 `Result/L/l2_features.csv` (L2 의 6-특징 배터리).

- SMD·방향무관 AUC·KS 로 균형표를 만들고, `|SMD| > 0.20` 또는 `AUC > 0.60` 이면 불균형
- `MATCH_METHOD` 로 정합 방법 하나를 사전 지정한다 — `pair`(유사 영상 짝지어 표집),
  `trim`(공통 범위 밖 제외), `none`
- 공변량 기록: 환자별 영상 수, 시퀀스 내 위치, 인접 등급 전이 여부, 환자 평균 등급.
  영상 품질 지표는 별도 산출물이 없어 1차 강도(mean·entropy)를 프록시로 넣었다

산출: `p2a_precheck_balance.csv`, `p2a_precheck_covariates.csv`,
`p2a_precheck_matched.csv`, `p2a_precheck_summary.json`

## 2. §3.2 대상군 · §3.4 mini-sequence

`build_minisequence.py` — `CXR/Merged` 만 읽는다. 영상 파일은 만들지도 복사하지도 않는다.

| 산출 | 내용 |
|---|---|
| `p2a_reading_sheet.csv` | 판독자용. 연도·기존등급·대상군·중복여부 모두 가림 |
| `p2a_admin_key.csv` | 분석용. 위 정보 + §3.4 기록 항목 |
| `p2a_image_manifest.csv` | `case_id` → Merged 원본 경로 |
| `p2a_summary.json` | 대상군별 영상·환자 수 |

주요 상수: `SHORT_RUN_MAX`, `MAX_MINISEQ`, `CONTROL_SEQ_LEN`, `DUPLICATE_FRAC`,
`MIN_DUP_GAP`, `TARGET_FULL_ENUMERATION`, `SEED`.

**§3.2 와 §3.4 가 충돌한다.** §3.2 는 표적군을 "203 ROI 전부" 로 규정하는데 §3.4 는
3·4 영상이 6장 이상인 환자에서 대표영상만 뽑으라고 한다. 2024 기준 45명 중 17명이
141장을 쥐고 있어 §3.4 를 지키면 203 장이 될 수 없다. `TARGET_FULL_ENUMERATION`
으로 어느 조항을 따를지 정한다 (기본 `False` = §3.4 우선).

판독시트는 `C00042.png` 형태의 blind 파일명만 담는다. 원본 `uid`·경로는 `24_`/`26_`
접두어로 연도를 노출하므로(§3.1 조건 1) 판독 폴더를 만들 때 manifest 대로 이름을 바꾼다.
`case_id` 도 선정 순서가 아니라 셔플된 최종 판독 순서로 매긴다.

hidden duplicate 는 낱장이 아니라 **mini-sequence 단위**로 재제시하고 새 blind 환자
ID 를 준다. 새 영상 파일이 아니라 같은 Merged 원본을 다른 `case_id` 로 한 번 더
가리킬 뿐이다 (manifest 에서 `source_path` 가 같고 `duplicate_flag=1`).

## 3. §4.4 검정력

`power_simulation.py` — `p2a_admin_key.csv` 의 실제 환자·영상 구조 위에서 재판독
등급을 생성하고, 공동 1차 평가변수의 단측 하한이 0 을 넘는 비율을 격자로 낸다.

- `C_year = S_upper,2024 − S_upper,2026`
- `C_cutpoint = S_upper,2024 − S_lower,2024`
- `C_noise = S_upper,2024 − S_dup` (`REQUIRE_C_NOISE`)

`rates_clipped = 1` 인 행은 이동률이 [0, 0.95] 로 잘려 목표 C 를 달성하지 못한
시나리오다. 검정력 곡선에서 제외하거나 기저 이동률을 조정한다.

**기저 이동률은 임시 자리값이다.** plan §4.4 는 예상 이동률의 출처(Phase 1 L4
자기일치인지 상위 경계 분리도인지)를 판독 전에 명시하도록 요구한다. 확정 전에는
단일 값이 아니라 격자 전체의 검정력 곡선으로 보고한다.

산출: `p2a_power_grid.csv`, `p2a_power_summary.json`

## 아직 없는 것

- §3.7 holdout 잠금 — 2026 미라벨 10명·126장을 명시적으로 제외·기록하는 절차
- 판독 결과 수거 후의 분석 스크립트 (§4.3 보조 평가변수, §4.5 민감도 분석)
