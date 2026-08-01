# -*- coding: utf-8 -*-
r"""base_rates_from_l4.py — 경계별 기저 이동률과 잡음 하 C 계열 값 (plan_phase2A_v2 §4.4).

기저 이동률 0.155 는 L4 의 RB 자기일치 ACC 0.6891 을 **전 등급에 걸쳐** 집계한 값에서
유도했다. 3/4 경계 사례는 원래 가장 안 맞으므로 전 등급 평균을 경계 기저율로 쓰면
시뮬레이션이 낙관 쪽으로 치우친다. 여기서는 재측정 세션쌍에서 §4.1 의 네 확률을
**경계별로 직접** 센다.

  p3↑ = P(재판독 = 4 | 기존 3)      p3↓ = P(재판독 ≤ 2 | 기존 3)
  p4↓ = P(재판독 ≤ 3 | 기존 4)      p2↑ = P(재판독 ≥ 3 | 기존 2)

같은 자료로 "기준 이동이 0 일 때 C 계열이 얼마인가" 도 낸다. 재측정은 같은 판독자가
같은 영상을 다시 본 것이므로 기준 드리프트가 없다고 보는 조건이며, 여기서 나오는
C_cutpoint 가 곧 순수 반복 판독 변동이 만드는 인공물의 크기다.

  S_upper = 0.5·p3↑ − 0.5·p4↓        S_lower = 0.5·p2↑ − 0.5·p3↓
  C_cutpoint = S_upper − S_lower = 0.5(p3↑ + p3↓) − 0.5(p4↓ + p2↑)
  A₃ = p3↑ − p3↓                     ← 등급 3 순 비대칭. 대칭 잡음이면 0.

C_cutpoint 는 기존 등급 3 의 양방향 이동을 **둘 다 더한다.** 등급 3 만 유독 불안정하면
드리프트가 없어도 C_cutpoint 가 양수로 뜬다. A₃ 는 그 경우 0 근처이므로 두 기전을 가른다.

두 출처를 쓴다. 재측정 8케이스만으로는 bootstrap 클러스터가 8개뿐이라 CI 가 신뢰
구간 노릇을 못 한다.

  4_재측정   같은 스터디를 다시 판독. 세션 간 영상·순서가 완전히 같아 위치로 대응된다
             (전 케이스에서 inference 벡터가 세션 간 완전 일치함을 확인).
  3_부분겹침  서로 다른 두 스터디가 같은 프레임을 각각 판독. 파일에 영상 식별자가 없고
             세션 길이도 달라 위치 대응이 불가하므로, RAW jpg 내용 서명으로 맞춘다
             (eda_report_v3.ipynb §3.5 와 같은 방식: 128x128 z-정규화, 평균절대차 < 0.10).

데이터: <cxr_root>/4_재측정/<case>/result{1,2,3}.txt
        <cxr_root>/3_부분겹침/<case>/result{n}.txt + RAW_IMAGE{n}/*.jpg
산출  : Result/P2A/p2a_base_rates.csv · p2a_base_rates_summary.json

CLI 대신 아래 상수를 직접 수정한다.
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent.parent
CXR_ROOT = "/shared/home/mai/JeongGeon/Private/CXR/2026 CXR"
REMEASURE = "4_재측정"
OVERLAP = "3_부분겹침"
OUT_DIR = REPO / "Result" / "P2A"

QUAD = ["RT", "LT", "RB", "LB"]
EDT = [f"user_edit_{r}" for r in QUAD]
TARGET_ROI = "RB"
N_BOOT = 5000
ALPHA = 0.05          # 양측 95% CI
SEED = 20260801

# 부분겹침 프레임 동일성 판정. 바이트 해시는 JPEG 재인코딩된 같은 프레임을 놓친다.
SIG_SIZE = 128
SIG_THR = 0.10

# 기준 출처. "재측정" · "부분겹침" · "합산" 중 하나.
# 기본값이 재측정인 이유: 부분겹침 공유 프레임에는 RB 기존 등급 4 가 하나도 없어
# p4↓ 를 추정할 수 없고, 따라서 3/4 경계(표적)에 대해 아무 정보도 주지 못한다.
# 합산하면 클러스터는 8→14 로 늘지만 등급 3 이 더 순한 표본으로 희석된다.
PRIMARY_SOURCE = "재측정"


def build_pairs(remeasure_dir):
    """케이스별 result*.txt → base 세션 vs 각 후속 세션 쌍. 케이스 id 를 함께 돌려준다.

    L4 와 같은 규칙(base = 첫 세션, min 길이 truncate)이되, 케이스 단위 bootstrap 을
    하려면 어느 쌍이 어느 케이스에서 왔는지 알아야 하므로 id 를 유지한다.
    """
    cases = sorted(d for d in os.listdir(remeasure_dir)
                   if os.path.isdir(os.path.join(remeasure_dir, d)))
    PA, PB, CID = [], [], []
    for case in cases:
        files = sorted(glob.glob(os.path.join(remeasure_dir, case, "result*.txt")))
        if len(files) < 2:
            continue
        labs = [pd.read_csv(f)[EDT].to_numpy() for f in files]
        base = labs[0]
        for l in labs[1:]:
            n = min(len(base), len(l))
            PA.append(base[:n]); PB.append(l[:n])
            CID.append(np.full(n, case))
    if not PA:
        raise SystemExit(f"재측정 쌍 없음: {remeasure_dir}")
    return np.vstack(PA), np.vstack(PB), np.concatenate(CID)


def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _sig(path):
    a = np.asarray(Image.open(path).convert("L").resize((SIG_SIZE, SIG_SIZE), Image.BILINEAR),
                   dtype=np.float32)
    return ((a - a.mean()) / (a.std() + 1e-6)).ravel()


def build_pairs_overlap(overlap_dir):
    """부분겹침: 세션 간 공유 프레임을 영상 서명으로 찾아 라벨 쌍을 만든다.

    result{n}.txt 의 행 순서는 같은 세션의 RAW_IMAGE{n}/*.jpg 자연정렬 순서와 대응한다.
    세션 사이에는 대응이 없으므로 서명 거리로 1:1 매칭한다(그리디, 임계 SIG_THR).
    """
    PA, PB, CID = [], [], []
    n_shared = 0
    for case in sorted(os.listdir(overlap_dir), key=_natkey):
        cp = os.path.join(overlap_dir, case)
        if not os.path.isdir(cp):
            continue
        units = []
        for f in sorted([x for x in os.listdir(cp) if re.match(r"result\d*\.txt$", x)],
                        key=_natkey):
            idx = re.search(r"result(\d*)", f).group(1)
            rd = os.path.join(cp, f"RAW_IMAGE{idx}")
            if not os.path.isdir(rd):
                rd = os.path.join(cp, "RAW_IMAGE")
            if not os.path.isdir(rd):
                continue
            jpg = sorted([x for x in os.listdir(rd) if x.lower().endswith(".jpg")], key=_natkey)
            lab = pd.read_csv(os.path.join(cp, f))[EDT].to_numpy()
            n = min(len(jpg), len(lab))
            if n == 0:
                continue
            units.append({"sig": np.stack([_sig(os.path.join(rd, x)) for x in jpg[:n]]),
                          "lab": lab[:n]})
        # 케이스당 **첫 두 세션 한 쌍만** 쓴다. 세션이 4개인 케이스(0427A_0428A)에서
        # 모든 조합을 쓰면 그 케이스가 전체 쌍의 절반을 차지해 추정이 한 환자에 지배된다.
        # 한 쌍만 쓰면 공유 프레임 69쌍으로, 라벨퀄리티_정합분석_보고서의 수와 일치한다.
        if len(units) < 2:
            continue
        A, B = units[0], units[1]
        d = np.abs(A["sig"][:, None, :] - B["sig"][None, :, :]).mean(-1)
        used = set()
        for a in range(d.shape[0]):
            b = int(np.argmin(d[a]))
            if d[a, b] < SIG_THR and b not in used:
                used.add(b)
                PA.append(A["lab"][a]); PB.append(B["lab"][b]); CID.append(case)
                n_shared += 1
    if not PA:
        return (np.empty((0, 4), int), np.empty((0, 4), int), np.array([], dtype=object)), 0
    return (np.vstack(PA), np.vstack(PB), np.array(CID, dtype=object)), n_shared


def rates(a, b):
    """§4.1 네 확률 + 파생값. 분모가 0 이면 nan."""
    def p(mask, cond):
        n = int(mask.sum())
        return (float(cond[mask].mean()) if n else float("nan")), n

    p3up, n3 = p(a == 3, b == 4)
    p3dn, _ = p(a == 3, b <= 2)
    p4dn, n4 = p(a == 4, b <= 3)
    p2up, n2 = p(a == 2, b >= 3)
    s_up = 0.5 * p3up - 0.5 * p4dn
    s_lo = 0.5 * p2up - 0.5 * p3dn
    return {"p3up": p3up, "p3down": p3dn, "p4down": p4dn, "p2up": p2up,
            "S_upper": s_up, "S_lower": s_lo, "C_cutpoint": s_up - s_lo,
            "A3": p3up - p3dn,
            "n_grade3": n3, "n_grade4": n4, "n_grade2": n2}


def boot_ci(a, b, cid, keys, rng):
    """케이스 단위 bootstrap. 같은 케이스의 모든 영상을 함께 재표집한다."""
    cases = np.unique(cid)
    idx_by_case = {c: np.where(cid == c)[0] for c in cases}
    acc = {k: [] for k in keys}
    for _ in range(N_BOOT):
        pick = rng.choice(cases, size=len(cases), replace=True)
        sel = np.concatenate([idx_by_case[c] for c in pick])
        r = rates(a[sel], b[sel])
        for k in keys:
            if not np.isnan(r[k]):
                acc[k].append(r[k])
    out = {}
    for k in keys:
        v = np.sort(acc[k])
        out[k] = ((float(np.quantile(v, ALPHA / 2)), float(np.quantile(v, 1 - ALPHA / 2)))
                  if v.size else (float("nan"), float("nan")))
    return out


KEYS = ["p3up", "p3down", "p4down", "p2up", "S_upper", "S_lower", "C_cutpoint", "A3"]


def analyse(PA, PB, CID, source, rng):
    out, detail = [], {}
    for k, roi in enumerate(QUAD):
        a, b = PA[:, k], PB[:, k]
        r = rates(a, b)
        ci = boot_ci(a, b, CID, KEYS, rng)
        detail[roi] = {**r, **{f"{x}_ci": list(ci[x]) for x in KEYS}}
        out.append({"source": source, "roi": roi,
                    **{x: round(r[x], 4) for x in KEYS},
                    **{f"{x}_lo": round(ci[x][0], 4) for x in KEYS},
                    **{f"{x}_hi": round(ci[x][1], 4) for x in KEYS},
                    "n_grade2": r["n_grade2"], "n_grade3": r["n_grade3"],
                    "n_grade4": r["n_grade4"],
                    "n_pairs": int(len(PA)), "n_cases": int(len(np.unique(CID)))})
    return out, detail


def show(rows, title):
    print(f"\n  [{title}]")
    print(f"    {'ROI':<5}{'p3↑':>9}{'p3↓':>9}{'p4↓':>9}{'p2↑':>9}{'n(2/3/4)':>16}")
    for r in rows:
        ns = "{}/{}/{}".format(r["n_grade2"], r["n_grade3"], r["n_grade4"])
        print(f"    {r['roi']:<5}{r['p3up']:>9.3f}{r['p3down']:>9.3f}"
              f"{r['p4down']:>9.3f}{r['p2up']:>9.3f}{ns:>16}")
    print(f"    {'ROI':<5}{'S_upper':>10}{'S_lower':>10}"
          f"{'C_cutpoint [95% CI]':>28}{'A₃ [95% CI]':>26}")
    for r in rows:
        cc = f"{r['C_cutpoint']:+.3f} [{r['C_cutpoint_lo']:+.3f}, {r['C_cutpoint_hi']:+.3f}]"
        a3 = f"{r['A3']:+.3f} [{r['A3_lo']:+.3f}, {r['A3_hi']:+.3f}]"
        print(f"    {r['roi']:<5}{r['S_upper']:>+10.3f}{r['S_lower']:>+10.3f}{cc:>28}{a3:>26}")


def main():
    rng = np.random.default_rng(SEED)
    rm_dir = os.path.join(CXR_ROOT, REMEASURE)
    ov_dir = os.path.join(CXR_ROOT, OVERLAP)
    if not os.path.isdir(rm_dir):
        raise SystemExit(f"재측정 폴더 없음: {rm_dir}  — CXR_ROOT 를 서버 실경로로 맞춘다")

    PA, PB, CID = build_pairs(rm_dir)
    print("=" * 78)
    print("base_rates_from_l4 — 경계별 기저 이동률 (기준 드리프트 0 조건)")
    print("=" * 78)
    print(f"  재측정   {len(np.unique(CID))}케이스 · {len(PA)}쌍  (세션 간 위치 대응)")

    rows, detail = [], {}
    r_rm, d_rm = analyse(PA, PB, CID, "재측정", rng)
    rows += r_rm; detail["재측정"] = d_rm

    if os.path.isdir(ov_dir):
        (OA, OB, OCID), n_sh = build_pairs_overlap(ov_dir)
        print(f"  부분겹침 {len(np.unique(OCID))}케이스 · {n_sh}쌍  "
              f"(영상 서명 매칭 {SIG_SIZE}px, 임계 {SIG_THR})")
        if len(OA):
            r_ov, d_ov = analyse(OA, OB, OCID, "부분겹침", rng)
            rows += r_ov; detail["부분겹침"] = d_ov
            # 두 출처를 합쳐 클러스터 수를 늘린다. 케이스 id 는 출처 접두어로 구분한다.
            CA = np.concatenate([PA, OA]); CB = np.concatenate([PB, OB])
            CC = np.concatenate([np.char.add("rm_", CID.astype(str)),
                                 np.char.add("ov_", OCID.astype(str))])
            r_all, d_all = analyse(CA, CB, CC, "합산", rng)
            rows += r_all; detail["합산"] = d_all
            print(f"  합산     {len(np.unique(CC))}케이스 · {len(CA)}쌍")
    else:
        print(f"  부분겹침 폴더 없음: {ov_dir} — 재측정만 사용")

    for src in ("재측정", "부분겹침", "합산"):
        sub = [r for r in rows if r["source"] == src]
        if sub:
            show(sub, f"{src} — 경계별 이동률과 드리프트 0 조건의 C 계열")

    print(f"\n  [{TARGET_ROI} 등급 구성 비교]  경계 추정에 쓸 수 있는 표본인지 확인")
    for r in [x for x in rows if x["roi"] == TARGET_ROI]:
        flag = "  ← 등급4 없음. 3/4 경계 추정 불가" if r["n_grade4"] == 0 else ""
        print(f"    {r['source']:<6} 등급2 {r['n_grade2']:>3} · 등급3 {r['n_grade3']:>3} · "
              f"등급4 {r['n_grade4']:>3}   ({r['n_cases']}케이스 {r['n_pairs']}쌍){flag}")

    main_src = PRIMARY_SOURCE if any(r["source"] == PRIMARY_SOURCE for r in rows) else "재측정"
    t = next(r for r in rows if r["source"] == main_src and r["roi"] == TARGET_ROI)
    print(f"\n  기준 출처: {main_src}  (PRIMARY_SOURCE 로 변경)")
    print(f"\n  [{TARGET_ROI} 해석]")
    print(f"    경계 기저 이동률 p3↑ {t['p3up']:.3f} · p3↓ {t['p3down']:.3f} · "
          f"p4↓ {t['p4down']:.3f} · p2↑ {t['p2up']:.3f}")
    print(f"    → power_simulation.py 의 BASE_MOVE 를 이 값들로 대체한다. "
          f"전 등급 평균 0.155 는 경계 기저율의 하한이다.")
    print(f"    드리프트 0 인데 C_cutpoint = {t['C_cutpoint']:+.3f} "
          f"[{t['C_cutpoint_lo']:+.3f}, {t['C_cutpoint_hi']:+.3f}]")
    if t["C_cutpoint_lo"] > 0:
        print(f"    ** CI 가 0 을 배제한다 — 등급 3 불안정이 만드는 인공물이 실재한다. **")
        print(f"       d_min=0.10 대비 {t['C_cutpoint'] / 0.10 * 100:.0f}% 수준이므로 "
              f"C_cutpoint 문턱을 이 값 위로 올리거나 A₃ 조건을 함께 걸어야 한다.")
    else:
        print(f"    CI 가 0 을 포함한다 — 대칭 잡음 가정과 모순되지 않는다.")
    print(f"    A₃ = {t['A3']:+.3f} [{t['A3_lo']:+.3f}, {t['A3_hi']:+.3f}] "
          f"(대칭 잡음이면 0, 상위 경계 이동이면 양수)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "p2a_base_rates.csv", index=False,
                              encoding="utf-8-sig")
    (OUT_DIR / "p2a_base_rates_summary.json").write_text(json.dumps({
        "cxr_root": CXR_ROOT,
        "sources": {"재측정": REMEASURE, "부분겹침": OVERLAP},
        "primary_source": main_src,
        "sig_match": {"size": SIG_SIZE, "threshold": SIG_THR},
        "n_boot": N_BOOT,
        "note": ("재측정 세션쌍 = 같은 판독자·같은 영상이므로 기준 드리프트 0 조건이다. "
                 "여기서 나오는 C_cutpoint 가 순수 반복 판독 변동이 만드는 인공물의 크기이며, "
                 "A₃ 는 등급 3 불안정(대칭, A₃≈0)과 상위 경계 이동(A₃>0)을 가른다."),
        "by_roi": detail,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {OUT_DIR}")
    for n in ("p2a_base_rates.csv", "p2a_base_rates_summary.json"):
        print(f"  - {n}")
    print("=" * 78)


if __name__ == "__main__":
    main()
