# -*- coding: utf-8 -*-
r"""power_simulation.py — Phase 2A 시뮬레이션 기반 검정력 분석 (plan_phase2a.md §4.4).

phase2a_admin_key.csv 의 실제 환자·영상 구조 위에서 재판독 등급을 생성하고,
공동 1차 평가변수의 검정력을 격자로 낸다.

  C_year     = S_upper,2024 − S_upper,2026
  C_cutpoint = S_upper,2024 − S_lower,2024
  (보조) C_noise = S_upper,2024 − S_dup   ← §6 진행 판정의 추가 조건

환자 단위 bootstrap 은 환자별 (합, 개수) 로 미리 접은 뒤 다항추출로 벡터화한다.
환자를 통째로 재표집하는 것과 결과가 같고, 파이썬 루프보다 수백 배 빠르다.

CLI 대신 아래 상수를 직접 수정한다.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "Result" / "P2A"
ADMIN_KEY = OUT_DIR / "p2a_admin_key.csv"

N_SIM = 1000
N_BOOT = 5000              # plan §4.5 최소 5,000회
SEED = 20260729

# ── 판정 규칙 ──
# plan §4.2 는 단측 검정이므로 단측 하한(5% 분위수)을 쓴다. 양측 2.5% 하한은 과보수적이다.
ALPHA = 0.05
NULL_MARGIN = 0.0
REQUIRE_C_NOISE = True     # §6 "C_noise 가 양수" 를 진행 판정에 함께 요구

# ── 시나리오 격자 (참값) ──
C_YEAR_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
C_CUTPOINT_GRID = (0.05, 0.10, 0.15, 0.20)

# ── 기저 이동률 — 출처: Phase 1 L4 판독자 자기일치 (plan §4.4 요구) ──
# Result/L/l4_reproducibility.csv 의 RB: 자기일치 ACC 0.6891 · MAE 0.3361.
#   불일치율      = 1 − 0.6891 = 0.3109
#   평균 이동 크기 = 0.3361 / 0.3109 = 1.08  → 불일치는 거의 전부 인접 등급 이동
#   방향 대칭 가정 → 한 방향 기저 이동률 = 0.3109 / 2 = 0.1555
# 이 값은 판독 기준 차이가 없어도 발생하는 반복 판독 변동의 바닥이다.
BASE_MOVE = 0.155
P26_3UP = BASE_MOVE        # 2026 기존 3 → 재판독 4
P26_4DOWN = BASE_MOVE      # 2026 기존 4 → 재판독 ≤3
P24_4DOWN = BASE_MOVE      # 2024 기존 4 → 재판독 ≤3
P24_3DOWN_LOWER = BASE_MOVE  # 2024 기존 3 → 재판독 ≤2

PATIENT_SIGMA = 0.35       # 환자 내 상관: 로짓 척도 환자 랜덤효과 SD
NON_EVALUABLE = 0.02       # 판독 불가 비율

# 환자별 통계 벡터 배치: (표적상향, 표적하향, 연도대조상향, 연도대조하향,
#                        하위상향, 하위하향, 중복상향, 중복하향) × (합, 개수)
BLOCKS = ("t_up", "t_dn", "y_up", "y_dn", "l_up", "l_dn", "d_up", "d_dn")
NCOL = 2 * len(BLOCKS)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def scenario_rates(c_year, c_cutpoint):
    """목표 C 값을 내는 이동률. S_upper = 0.5·p_up − 0.5·p_down 을 역산한다."""
    s26 = 0.5 * P26_3UP - 0.5 * P26_4DOWN
    s24 = s26 + c_year
    s_low = s24 - c_cutpoint
    raw_3up, raw_2up = 2 * s24 + P24_4DOWN, 2 * s_low + P24_3DOWN_LOWER
    r = {"p26_3up": P26_3UP, "p26_4down": P26_4DOWN, "p24_4down": P24_4DOWN,
         "p24_3down_lower": P24_3DOWN_LOWER,
         "p24_3up": float(np.clip(raw_3up, 0.0, 0.95)),
         "p24_2up": float(np.clip(raw_2up, 0.0, 0.95))}
    # 클리핑되면 목표 C 를 달성하지 못하므로 조용히 넘기지 않는다.
    r["_clipped"] = int(abs(r["p24_3up"] - raw_3up) > 1e-9 or abs(r["p24_2up"] - raw_2up) > 1e-9)
    return r


def move_probs(year, grade, rates):
    """(상향확률, 하향확률). 순서형이므로 한 번만 추첨해 파생시킨다."""
    if year == "2024" and grade == 3:
        return rates["p24_3up"], rates["p24_3down_lower"]
    if year == "2024" and grade == 4:
        return 0.0, rates["p24_4down"]
    if year == "2024" and grade == 2:
        return rates["p24_2up"], 0.0
    if year == "2026" and grade == 3:
        return rates["p26_3up"], 0.0
    if year == "2026" and grade == 4:
        return 0.0, rates["p26_4down"]
    return 0.0, 0.0


def draw_reread(rows, rates, rng):
    """case_id → 재판독 등급 (판독 불가는 None).

    상향·하향을 독립 베르누이로 뽑으면 한 영상이 '4로 상향'과 '2 이하로 하향'을 동시에
    만족할 수 있다. 균등난수 하나로 세 갈래(상향/유지/하향)를 가른다.
    환자 랜덤효과는 잠재 이동이므로 상향 로짓에 +shift, 하향 로짓에 −shift 로 넣는다.
    """
    shift = {}
    for r in rows:
        if r["patient_id"] not in shift:
            shift[r["patient_id"]] = rng.normal(0.0, PATIENT_SIGMA)

    n = len(rows)
    u = rng.random(n)
    ev = rng.random(n) >= NON_EVALUABLE
    out = {}
    for k, r in enumerate(rows):
        g = r["_g"]
        if g is None or not ev[k]:
            out[r["case_id"]] = None
            continue
        p_up, p_dn = move_probs(r["year"], g, rates)
        s = shift[r["patient_id"]]
        p_up = float(inv_logit(logit(p_up) + s)) if p_up > 0 else 0.0
        p_dn = float(inv_logit(logit(p_dn) - s)) if p_dn > 0 else 0.0
        out[r["case_id"]] = min(4, g + 1) if u[k] < p_up else \
                            max(0, g - 1) if u[k] < p_up + p_dn else g
    return out


def patient_stats(pat_index, groups, pairs, reread):
    """환자 × (합, 개수) 행렬. bootstrap 은 이 행렬의 행을 재표집하는 것과 같다."""
    M = np.zeros((len(pat_index), NCOL))
    col = {b: 2 * i for i, b in enumerate(BLOCKS)}

    def put(pid, block, hit):
        j = col[block]
        M[pat_index[pid], j] += hit
        M[pat_index[pid], j + 1] += 1

    for name, up_blk, dn_blk in (("target", "t_up", "t_dn"),
                                 ("year_ctl", "y_up", "y_dn"),
                                 ("lower_ctl", "l_up", "l_dn")):
        lower = name == "lower_ctl"
        for r in groups[name]:
            v = reread[r["case_id"]]
            if v is None:
                continue
            if lower:
                if r["_g"] == 2: put(r["patient_id"], up_blk, int(v >= 3))
                elif r["_g"] == 3: put(r["patient_id"], dn_blk, int(v <= 2))
            else:
                if r["_g"] == 3: put(r["patient_id"], up_blk, int(v == 4))
                elif r["_g"] == 4: put(r["patient_id"], dn_blk, int(v <= 3))

    for pid, base, dup in pairs:
        a, b = reread.get(base), reread.get(dup)
        if a is None or b is None:
            continue
        if a == 3: put(pid, "d_up", int(b == 4))
        elif a == 4: put(pid, "d_dn", int(b <= 3))
    return M


def contrasts_from(M):
    """M: (..., NCOL) 누적합 → 대비값. 개수 0 이면 NaN."""
    s = M[..., 0::2]
    c = M[..., 1::2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(c > 0, s / np.where(c > 0, c, 1), np.nan)
    t_up, t_dn, y_up, y_dn, l_up, l_dn, d_up, d_dn = (p[..., i] for i in range(len(BLOCKS)))
    su24 = 0.5 * t_up - 0.5 * t_dn
    su26 = 0.5 * y_up - 0.5 * y_dn
    slo = 0.5 * l_up - 0.5 * l_dn
    sdup = 0.5 * d_up - 0.5 * d_dn
    return {"S_upper_2024": su24, "S_upper_2026": su26, "S_lower_2024": slo, "S_dup": sdup,
            "C_year": su24 - su26, "C_cutpoint": su24 - slo, "C_noise": su24 - sdup}


def bootstrap_lower(M, keys, rng):
    """환자 단위 stratified bootstrap 단측 하한 (§4.5)."""
    n = M.shape[0]
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)   # (N_BOOT, n_pat)
    c = contrasts_from(w @ M)
    out = {}
    for k in keys:
        v = c[k][~np.isnan(c[k])]
        out[k] = float(np.quantile(v, ALPHA)) if v.size else float("nan")
    return out


def main():
    rng = np.random.default_rng(SEED)
    rows = read_csv(ADMIN_KEY)
    if not rows:
        raise SystemExit(f"admin key is empty: {ADMIN_KEY}")
    for r in rows:
        try: r["_g"] = int(float(r["orig_RB"]))
        except Exception: r["_g"] = None

    base = [r for r in rows if not int(r.get("duplicate_flag", 0))]
    # hidden duplicate 는 같은 uid 의 재제시이므로 uid 로 원본과 짝짓는다.
    by_uid = defaultdict(list)
    for r in rows:
        by_uid[r["uid"]].append(r)
    pairs = []
    for v in by_uid.values():
        first = [x for x in v if not int(x.get("duplicate_flag", 0))]
        rep = [x for x in v if int(x.get("duplicate_flag", 0))]
        if len(first) == 1 and len(rep) == 1:
            pairs.append((first[0]["patient_id"], first[0]["case_id"], rep[0]["case_id"]))

    groups = {"target": [r for r in base if int(r.get("target_2024_rb_upper", 0))],
              "year_ctl": [r for r in base if int(r.get("control_2026_rb_upper", 0))],
              "lower_ctl": [r for r in base if int(r.get("control_2024_rb_lower", 0))]}
    pats = sorted({r["patient_id"] for r in base})
    pat_index = {p: i for i, p in enumerate(pats)}

    keys = ["C_year", "C_cutpoint"] + (["C_noise"] if REQUIRE_C_NOISE else [])
    W = 76
    print("=" * W)
    print("power_simulation — Phase 2A §4.4 검정력 (공동 1차 " + " · ".join(keys) + ")")
    print("=" * W)
    print(f"  영상 {len(base)}장 · 환자 {len(pats)}명 · 중복쌍 {len(pairs)}개 · "
          f"n_sim {N_SIM} · n_boot {N_BOOT}")
    print(f"\n  {'C_year':>8}{'C_cut':>8}{'power':>9}{'평균 C_year':>13}"
          f"{'평균 C_cut':>12}{'clip':>6}")

    out_rows = []
    for c_year in C_YEAR_GRID:
        for c_cut in C_CUTPOINT_GRID:
            rates = scenario_rates(c_year, c_cut)
            clipped = rates.pop("_clipped")
            hit, acc = 0, defaultdict(list)
            for _ in range(N_SIM):
                reread = draw_reread(rows, rates, rng)
                M = patient_stats(pat_index, groups, pairs, reread)
                pt = contrasts_from(M.sum(axis=0))
                for k, v in pt.items():
                    if not np.isnan(v):
                        acc[k].append(float(v))
                lo = bootstrap_lower(M, keys, rng)
                hit += int(all(lo[k] > NULL_MARGIN for k in keys))
            row = {"target_C_year": c_year, "target_C_cutpoint": c_cut,
                   "power": hit / N_SIM,
                   "mean_C_year": float(np.mean(acc["C_year"])) if acc["C_year"] else float("nan"),
                   "mean_C_cutpoint": float(np.mean(acc["C_cutpoint"])) if acc["C_cutpoint"] else float("nan"),
                   "mean_C_noise": float(np.mean(acc["C_noise"])) if acc["C_noise"] else float("nan"),
                   "rates_clipped": clipped,
                   **{k: rates[k] for k in sorted(rates)},
                   "n_sim": N_SIM, "n_boot": N_BOOT, "patient_sigma": PATIENT_SIGMA}
            out_rows.append(row)
            print(f"  {c_year:>8.2f}{c_cut:>8.2f}{row['power']:>9.3f}"
                  f"{row['mean_C_year']:>13.3f}{row['mean_C_cutpoint']:>12.3f}"
                  f"{'  X' if clipped else '   ':>6}")

    write_csv(OUT_DIR / "p2a_power_grid.csv", out_rows)
    (OUT_DIR / "p2a_power_summary.json").write_text(json.dumps({
        "admin_key": str(ADMIN_KEY),
        "n_images": len(base),
        "n_patients": len(pats),
        "n_duplicate_pairs": len(pairs),
        "success_rule": f"one-sided lower{int((1 - ALPHA) * 100)} > {NULL_MARGIN} for " + ", ".join(keys),
        "base_move_rate": BASE_MOVE,
        "rate_source_note": ("기저 이동률 0.155 = Phase 1 L4 RB 자기일치에서 유도 "
                             "(ACC 0.6891 → 불일치 0.3109, MAE 0.3361 로 인접 이동 확인, "
                             "방향 대칭 가정하여 절반). plan §4.4 의 출처 명시 요구를 충족한다."),
        "rows": out_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  clip=X 는 이동률이 [0, 0.95] 로 잘려 목표 C 를 달성 못한 시나리오다.")
    print(f"  기저 이동률 {BASE_MOVE} = Phase 1 L4 RB 자기일치(ACC 0.6891)에서 유도.")
    print("\n" + "=" * W)
    print(f"[save] {OUT_DIR}")
    for n in ("p2a_power_grid.csv", "p2a_power_summary.json"):
        print(f"  - {n}")
    print("=" * W)


if __name__ == "__main__":
    main()
