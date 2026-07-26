# 실험 계획서

영역 순서: **[RT, LT, RB, LB]** (전 실험 공통)

## 1. 데이터

| 항목 | 값 |
|---|---|
| 경로 | `CXR/Merged/` |
| 라벨 | `labels.csv` — `uid, patient_id, RT, LT, RB, LB, image_path` |
| 연도 구분 | `patient_id` 접두어 `24_` / `26_` |
| 이미지 | `images_normalize/<uid>.png` — 512×512 grayscale, seg+STN 정렬 완료 |
| 마스크 | `masks/<uid>.png` — 512×512, 정렬 완료 |
| 등급 | 0~4 (5-class), ROI 4개 |
| 2024 | 68 환자 / 718 영상 |
| 2026 | 46 환자 / 587 영상 |

## 2. 학습 규약 (공통)

| 파라미터 | 값 |
|---|---|
| epochs | 50 |
| early-stop patience | 10 (val MAE 기준) |
| 최종 test 모델 | **best-val 체크포인트 reload** |
| 백본 동결 | FREEZE_BLOCKS = 6 |
| train : val | **8 : 2** (val 20%, 환자 단위) — E1·E2 는 5-fold 비중복 val |
| batch size | 32 |
| optimizer | AdamW |
| lr | encoder 1e-5 / head 1e-4 |
| weight decay | encoder 5e-5 / head 5e-4 |
| scheduler | CosineAnnealing (T_max = 50) |
| 정규화 | Normalize([0.56], [0.17]) |
| seeds | 1, 2, 42 |

## 3. 모델

| 모델 | 구성 |
|---|---|
| **DORGA** | MRM ViT-base/16-512 백본(`DORGA_Brixia.pth`, in_chans=1, drop_path 0.1, global_pool avg) + Dynamic prior, R=4·C=5·K=7, proj_dim=768, GNN heads=4·dropout=0.1 |
| **BSNet** | ResNet18 + 하드어텐션, BrixiaLoss |
| **PAFE** | ResNet34+ViT (3ch 내부복제), CE |

## 4. E 실험

| 실험 | 설계 | 산출 |
|---|---|---|
| **E1** cross | **val 5-fold(8:2) 비중복.** 방향마다 train연도를 5-fold → fold k=val(20%)·나머지=train(80%), **test=반대연도 전체(고정)**. split 1~5. fwd(2024→2026)·rev(2026→2024). δ_obs=(rev−fwd)/2, γ=(rev+fwd)/2, split 평균 | `checkpoint/E1/dorga/{24to26,26to24}_split{1..5}/` |
| **E2** in-domain raw | 코호트별 환자 5-fold(전 환자 1회 test) × init_seeds 42·1·2, **fold 내 8:2 val**. **raw**(전 환자). Δg_raw = g₂₀₂₄ − g₂₀₂₆ | `checkpoint/E2/dorga/` + `e2_summary.json` |
| **E3** in-domain matched | E2와 동일하나 **matched**(두 코호트 등급분포 정합 서브샘플)만. Δg_matched. 수축분 = Δg_raw − Δg_matched | `checkpoint/E3/dorga/` + `e3_summary.json` |
| **E4** 양성대조 | 한 코호트(2024)를 정합 두 반쪽 H1·H2. H2에 오프셋 β ∈ {0, 0.25, 0.5, 1.0} 주입. reps 42·1·2. 복원곡선(β→δ) 기울기·절편 | `checkpoint/E4/` + `e4_summary.json` |
| **E5** 음성대조 | β=0, 학습량 비대칭 train_frac ∈ {1.0, 0.5, 0.25}. reps 42·1·2. δ_spurious 누출 | `checkpoint/E5/` + `e5_summary.json` |

판정: δ_corr = δ_obs + Δg/2 (E3 matched). `A/run_all.py` → `A1_verdict.json`.

## 5. L 분석 (학습 무관: L1·L2·L4)

| 레벨 | 계산 | 산출 |
|---|---|---|
| **L1** 구조·분포 | 환자·영상수, 시퀀스길이, 평균등급, 등급 0~4 비율 (labels.csv) | `Result/L/l1_structure.csv` |
| **L2** 라벨-영상 | ROI **16-특징**[1차 4(mean·median·uniformity·entropy) + GLRLM 6(SRE·LRE·GLN·RLN·HGLRE·LRHGLE) + GLSZM 6(SAE·LZE·GLN·SZN·ZP·HGLZE)]. 그레이 Ng=16, ROI [min,max] 양자화, GLRLM 4방향 합산, GLSZM 8-연결. 인접등급 방향무관 AUC=max(a,1−a). 경계 3→4 | `Result/L/{l2_features,l2_auc}.csv` |
| **L3** 방향반전 분해 | E1 예측 → δ·γ (모델 의존) | (= E1) |
| **L4** 재현성 | 재측정 2세션 → 자기일치 ACC, weighted κ(quadratic), Wilcoxon | `Result/L/l4_reproducibility.csv` |

## 6. 평가·통계

| 항목 | 값 |
|---|---|
| 지표 | ACC, MAE, bias(pred−label) |
| bias 단위 | 환자 (연속 영상 상관 → 환자를 독립 단위) |
| CI | 환자 단위 부트스트랩, n=5000, 95% |
| 방향반전 | δ=(rev−fwd)/2, γ=(rev+fwd)/2 |
| A1 판정 | Δg CI가 0 배제 시 A1 위반 |

## 7. 실행

```bash
python Experiment/E/e1_cross.py                       # fwd·rev × split 1~5 (10 arm)
python Experiment/E/e1_cross.py --mode fwd --split 1  # 특정 방향·split 만
python Experiment/E/e2_indomain_raw.py --folds 5 --init_seeds 42 1 2      # E2 raw
python Experiment/E/e3_indomain_matched.py --folds 5 --init_seeds 42 1 2  # E3 matched
python Experiment/E/e4_positive_control.py --year 2024 --betas 0 0.25 0.5 1.0 --reps 42 1 2
python Experiment/E/e5_negative_control.py --year 2024 --fracs 1.0 0.5 0.25 --reps 42 1 2
python Experiment/A/run_all.py

python Experiment/L/l1_structure.py
python Experiment/L/l2_label_image.py
python Experiment/L/l4_reproducibility.py --rr <reread.csv>

python Experiment/E/summary.py
python check_value_l.py ; python check_value_e.py ; python check_value_a.py
```
