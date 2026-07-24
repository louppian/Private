# A1 검증 실험 코드 (draft.md §5.2 / A1_검증실험계획)

방향 반전 분해의 유일한 식별 가정 **A1(모델 오차 방향 대칭성, $g_f=g_r$)** 을 데이터로 검증한다.
δ 가 라벨 성분(판독 기준 이동)을 식별하려면 A1 이 성립해야 하며, 위반 시 δ 는
라벨 차이와 모델 반대칭분 $(g_r-g_f)/2$ 의 혼합이 된다.

## 구성

| 파일 | 역할 |
|---|---|
| `a1_common.py` | 코어. 기존 `dorga_train_2026to2024.py` 를 import 해 재사용하고 `make_split` 만 몽키패치. splitter·라벨주입·분포정합·δ/s 분해·bootstrap 유틸 |
| `e1_indomain_kfold.py` | **E1** 환자 k-fold in-domain → $g_{2024},g_{2026}$ 추정, $H_0:g_{2024}=g_{2026}$ 검정 |
| `e2_matched_indomain.py` | **E2** 분포 정합 in-domain → Δg 가 수축 기원인지 진짜 비대칭인지 분리 |
| `e3_positive_control.py` | **E3** 반합성 양성대조 → 알려진 오프셋 β 복원곡선(추정기 무편향 검증) |
| `e4_negative_control.py` | **E4** 반합성 음성대조 → 학습량 비대칭(β=0)이 δ 로 누출되는 양 정량화 |
| `run_all.py` | E3→E1→E2→E4 실행 + **δ 보정·최종 판정** (`runs/A1_verdict.txt`) |

## 전제

- 실행에 **DORGA 레포**(`C:\Code\DORGA`), 가중치(seg/stn/MRM), **GPU**, `torch_ev` 환경 필요 —
  모든 무거운 로직은 `dorga_train_2026to2024.py` 재사용이라 그 스크립트가 돌면 이 코드도 돈다.
- 데이터: `split_manifest.csv`(2024 718/68, 2026 587/46), 2024 디스크 마스크, 2026 in-code seg+STN 캐시.
- 산출: `D:\npjDM2026\runs\{E1..E4}\` arm별 `results.json`·`test_preds.npz`, `runs\A1_verdict.{json,txt}`.

## 실행

```bash
cd D:\npjDM2026\code
# 전체 (E3→E1→E2→E4→판정).  epochs 는 기존 recipe 와 동일 50 기본.
python run_all.py --epochs 50

# 개별 실행
python e3_positive_control.py --year 2024 --betas 0 0.25 0.5 1.0 --reps 42 1 2
python e1_indomain_kfold.py   --folds 5 --init_seeds 42 1 2
python e2_matched_indomain.py --folds 5 --init_seeds 42 1 2
python e4_negative_control.py --year 2024 --fracs 1.0 0.5 0.25

# 학습 없이 판정만 (요약 json 이 이미 있을 때)
python run_all.py --verdict_only
```

## 판정 규칙 (계획 §6)

| E1·E2 결과 | A1 | δ 처리 | RB·LT 결론 |
|---|---|---|---|
| Δg ≈ 0 (CI 0 포함, 반폭 작음) | 성립 | δ 그대로 신뢰 | 라벨 드리프트 확정 |
| Δg ≠ 0, E2 정합 후 소멸 | 조건부 | 수축분 제거 후 사용 | 대체로 유지 |
| Δg ≠ 0, E2 후 잔존 | 위반 | $δ_{corr}=δ_{obs}+Δg/2$ | 잔존 δ 만큼만 주장 |

`run_all.py` 는 기존 실데이터 방향반전 run(`dorga_bias_direction_runs/.../test_preds.npz`)에서
$δ_{obs}$ 를 ROI별로 재계산하고, E1(raw)·E2(matched) 의 Δg 로 각각 보정한
$δ_{corr}$ 표를 `A1_verdict.txt` 에 낸다. **최종 질문 = "RB δ +0.231 중 보정 후 얼마가 살아남는가".**

## 설계 메모 / 한계

- **E4 노브**는 영상 변형이 아니라 **학습 데이터량 비대칭**(split-only)이라 `InhaUHMaskDataset` 수정 불필요.
  영상 블러/노이즈 변형본이 필요하면 `a1_common` 의 dataset 훅을 추가해야 한다(선택).
- **E3 라벨 주입**은 순서형 stochastic offset + `clip(0,C-1)` — 큰 β 에서 천장효과로 복원 기울기가
  감쇠하며, 이는 복원곡선이 스스로 드러낸다(β≤0.5 권장 구간에서 판정).
- in-domain $g_c$(train c/test c) 와 cross $g_f$(train c/test d) 사이 **수축 갭**은 E2 정합으로 제거한다.
- 기존 bias run 은 단일 seed(s42)라 $δ_{obs}$ 는 점추정. 다seed 재현 시 `dorga_train_2026to2024.py --seeds 42 1 2` 로 갱신 후 `run_all.py --verdict_only`.
