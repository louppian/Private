# -*- coding: utf-8 -*-
r"""power_patient_axis.py — Phase 2A 필요 환자 수 (plan_phase2A_v2.md §4.4).

v2 §4.4 는 판독 전에 "1차 성공 판정에 필요한 환자 수"를 고정하라고 요구한다.
power_simulation.py 는 표본을 전수로 고정하고 참효과만 바꾸므로 그 값을 낼 수 없다.
이 스크립트는 판독 환자 수를 줄여가며 공동 1차 검정력 곡선을 그려, 목표 power 0.80
을 넘기는 최소 환자 수를 찾는다.

코호트가 유한하다 — 2024 RB 3·4 보유 45명, 2026 33명이 상한이라 늘릴 수 없다.
따라서 축소 방향으로만 스캔하며 frac=1.0 이 전수다. 층(코호트)별로 비율을 적용하고,
어느 환자가 뽑히는지는 복제마다 다시 뽑아 특정 환자 조합에 기댄 값이 나오지 않게 한다.

판정·모형은 power_simulation.py 를 그대로 쓴다(중복 구현하지 않는다).
  성공 = C_year·C_cutpoint 모두 bootstrap 단측 하한 > 0 그리고 점추정 ≥ d_min

입력: Result/P2A/p2a_admin_key.csv  (build_minisequence.py 산출)
산출: Result/P2A/p2a_power_patient_axis_grid.csv · _summary.json

CLI 대신 아래 상수를 직접 수정한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_simulation as ps  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

# ── 환자 수 축 ──
PATIENT_FRACTIONS = (0.4, 0.5, 0.6, 0.7, 0.85, 1.0)

# 효과크기는 목표 power 를 넘길 수 있는 구간만 본다. power_simulation.py 결과에서
# C_cutpoint < 0.15 는 어떤 C_year 에서도 0.80 에 못 미치므로 제외한다.
C_YEAR_GRID = (0.15, 0.20)
C_CUTPOINT_GRID = (0.15, 0.20)

N_SIM = 500
N_BOOT = 2000

TARGET_POWER = 0.80


def main():
    ps.PATIENT_FRACTIONS = PATIENT_FRACTIONS
    ps.C_YEAR_GRID = C_YEAR_GRID
    ps.C_CUTPOINT_GRID = C_CUTPOINT_GRID
    ps.N_SIM = N_SIM
    ps.N_BOOT = N_BOOT
    ps.OUT_PREFIX = "p2a_power_patient_axis"

    rows = ps.main()
    if not rows:
        return

    W = 76
    print("\n" + "=" * W)
    print(f"필요 환자 수 — 목표 power {TARGET_POWER:.2f} 를 넘기는 최소 표본")
    print("=" * W)
    print(f"  {'C_year':>7}{'C_cut':>7}{'필요 환자':>10}{'필요 영상':>10}{'그때 power':>12}")
    for cy in C_YEAR_GRID:
        for cc in C_CUTPOINT_GRID:
            cells = sorted((r for r in rows
                            if r["target_C_year"] == cy and r["target_C_cutpoint"] == cc),
                           key=lambda r: r["n_patients_used"])
            hit = next((r for r in cells if r["power"] >= TARGET_POWER), None)
            if hit is None:
                best = cells[-1]
                print(f"  {cy:>7.2f}{cc:>7.2f}{'미달':>10}{'':>10}"
                      f"{best['power']:>12.3f}  (전수 {best['n_patients_used']}명에서도 미달)")
            else:
                print(f"  {cy:>7.2f}{cc:>7.2f}{hit['n_patients_used']:>10}"
                      f"{hit['n_images_used']:>10.0f}{hit['power']:>12.3f}")
    print("\n  전수(frac=1.0)에서도 미달이면 v2 §4.4 대로 Phase 2A 를 확증적 검정이 아니라")
    print("  정밀도 중심의 표적 재판독으로 기술한다.")
    print("=" * W)


if __name__ == "__main__":
    main()
