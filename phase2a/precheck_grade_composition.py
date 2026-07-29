# -*- coding: utf-8 -*-
r"""precheck_grade_composition.py — Phase 2A §3.6 기존 등급 조건화 사전 점검.

Phase 2A 는 기존 등급으로 대상을 선정하고 이동을 계산한다. 같은 기존 등급 3 이라도
두 연도의 실제 중증도 구성이 다르면 판독 기준 차이가 없어도 이동률이 갈린다.
재판독 **전에** 두 연도의 RB 기존 등급 2·3·4 영상 특징 분포를 비교하고, 차이가 크면
사전 지정한 정합 방법 하나를 적용해 주 분석용 표본을 만든다.

입력: Result/L/l2_features.csv (L2 의 6-특징 배터리 산출물)
      없으면 Experiment/L/l2_label_image.py 를 먼저 돌린다.

산출: Result/P2A/p2a_precheck_balance.csv   특징 × 등급 균형표 (SMD·AUC·KS)
      Result/P2A/p2a_precheck_covariates.csv 영상별 공변량 (§3.6 기록 요구)
      Result/P2A/p2a_precheck_matched.csv   정합 표본 (주 분석용 uid 목록)
      Result/P2A/p2a_precheck_summary.json  판정과 적용한 방법

CLI 대신 아래 상수를 직접 수정한다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent.parent
FEATURES = REPO / "Result" / "L" / "l2_features.csv"
OUT_DIR = REPO / "Result" / "P2A"

TARGET_ROI = "RB"
GRADES = (2, 3, 4)              # §3.6: 등급 3·4, 필요 시 2
FEATS = ("mean", "median", "entropy", "uniformity", "glrlm_SRE", "glszm_SAE")

# ── 불균형 판정 기준 ──
SMD_THRESHOLD = 0.20            # |표준화 평균차| 이 값을 넘으면 불균형
AUC_THRESHOLD = 0.60            # 방향무관 AUC. 연도가 영상특징으로 갈리면 불균형

# ── 사전 지정 정합 방법 (하나만 고른다, §3.6) ──
#   "pair" 유사 영상 짝지어 표집 · "trim" 공통 범위 밖 제외 · "none" 정합 없음
MATCH_METHOD = "pair"
CALIPER = 1.0                   # pair: 표준화 6차원 거리 상한
TRIM_QUANTILE = 0.05            # trim: 각 특징의 공통 범위를 이 분위수로 자른다
SEED = 20260729


def load_features():
    if not FEATURES.exists():
        raise SystemExit(f"{FEATURES} 없음 — Experiment/L/l2_label_image.py 를 먼저 실행한다")
    df = pd.read_csv(FEATURES)
    cols = [f"{TARGET_ROI}_{f}" for f in FEATS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"특징 컬럼 없음: {missing}")
    return df


def covariates(df):
    """§3.6 기록 요구 — 환자별 영상 수, 시퀀스 내 위치, 등급 전이 전후, 환자 평균 등급.

    영상 품질 지표는 별도 산출물이 없어 1차 강도 통계(mean·entropy)로 대신 기록한다.
    """
    d = df.sort_values(["patient", "uid"]).copy()
    g = d[TARGET_ROI].to_numpy()
    pat = d["patient"].to_numpy()

    d["seq_pos"] = d.groupby("patient").cumcount() + 1
    d["patient_n_img"] = d.groupby("patient")["uid"].transform("size")
    d["seq_pos_frac"] = d["seq_pos"] / d["patient_n_img"]
    d["patient_mean_grade"] = d.groupby("patient")[TARGET_ROI].transform("mean")

    same_prev = np.r_[False, pat[1:] == pat[:-1]]
    same_next = np.r_[pat[:-1] == pat[1:], False]
    prev_diff = np.r_[False, g[1:] != g[:-1]] & same_prev
    next_diff = np.r_[g[:-1] != g[1:], False] & same_next
    d["adj_transition"] = (prev_diff | next_diff).astype(int)
    d["quality_proxy_mean"] = d[f"{TARGET_ROI}_mean"]
    d["quality_proxy_entropy"] = d[f"{TARGET_ROI}_entropy"]
    return d


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def auc_undirected(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = mannwhitneyu(b, a, alternative="two-sided").statistic
    x = u / (len(a) * len(b))
    return float(max(x, 1 - x))


def balance_table(d, tag):
    """연도 간 균형표. 2024 를 기준군, 2026 을 비교군으로 둔다."""
    rows = []
    for grade in GRADES:
        s = d[d[TARGET_ROI] == grade]
        a = s[s.year == 2024]
        b = s[s.year == 2026]
        for f in FEATS:
            col = f"{TARGET_ROI}_{f}"
            x, y = a[col].to_numpy(), b[col].to_numpy()
            p = ks_2samp(x[~np.isnan(x)], y[~np.isnan(y)]).pvalue if len(x) and len(y) else np.nan
            s_, u_ = smd(x, y), auc_undirected(x, y)
            rows.append({
                "sample": tag, "roi": TARGET_ROI, "grade": grade, "feature": f,
                "n_2024": len(a), "n_2026": len(b),
                "mean_2024": float(np.nanmean(x)) if len(x) else np.nan,
                "mean_2026": float(np.nanmean(y)) if len(y) else np.nan,
                "smd": s_, "auc": u_, "ks_p": float(p) if p == p else np.nan,
                "imbalanced": int((abs(s_) > SMD_THRESHOLD if s_ == s_ else False)
                                  or (u_ > AUC_THRESHOLD if u_ == u_ else False)),
            })
    return pd.DataFrame(rows)


def standardize(s, cols):
    x = s[cols].to_numpy(float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def match_pair(d, rng):
    """유사 영상 짝지어 표집. 등급 안에서 표준화 6차원 거리로 1:1 탐욕 매칭."""
    cols = [f"{TARGET_ROI}_{f}" for f in FEATS]
    keep = []
    for grade in GRADES:
        s = d[d[TARGET_ROI] == grade].copy()
        if s.empty:
            continue
        z = standardize(s, cols)
        is24 = (s.year == 2024).to_numpy()
        i24 = np.where(is24)[0]
        i26 = np.where(~is24)[0]
        if len(i24) == 0 or len(i26) == 0:
            continue
        # 적은 쪽을 기준으로 짝을 찾는다.
        base, other = (i24, i26) if len(i24) <= len(i26) else (i26, i24)
        order = rng.permutation(len(base))
        used = np.zeros(len(other), bool)
        uid = s["uid"].to_numpy()
        for bi in order:
            b = base[bi]
            dist = np.linalg.norm(z[other] - z[b], axis=1)
            dist[used] = np.inf
            j = int(np.argmin(dist))
            if dist[j] <= CALIPER:
                used[j] = True
                keep.extend([uid[b], uid[other[j]]])
    return set(keep)


def match_trim(d):
    """공통 범위 밖 영상 제외. 등급별로 두 연도 분포의 겹치는 구간만 남긴다."""
    cols = [f"{TARGET_ROI}_{f}" for f in FEATS]
    keep = set()
    for grade in GRADES:
        s = d[d[TARGET_ROI] == grade]
        if s.empty:
            continue
        ok = np.ones(len(s), bool)
        for c in cols:
            a = s.loc[s.year == 2024, c].to_numpy(float)
            b = s.loc[s.year == 2026, c].to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                continue
            lo = max(np.nanquantile(a, TRIM_QUANTILE), np.nanquantile(b, TRIM_QUANTILE))
            hi = min(np.nanquantile(a, 1 - TRIM_QUANTILE), np.nanquantile(b, 1 - TRIM_QUANTILE))
            v = s[c].to_numpy(float)
            ok &= (v >= lo) & (v <= hi)
        keep |= set(s.loc[ok, "uid"])
    return keep


def main():
    rng = np.random.default_rng(SEED)
    d = covariates(load_features())

    raw = balance_table(d, "raw")
    n_bad = int(raw.imbalanced.sum())

    if MATCH_METHOD == "pair":
        kept = match_pair(d, rng)
    elif MATCH_METHOD == "trim":
        kept = match_trim(d)
    elif MATCH_METHOD == "none":
        kept = set(d["uid"])
    else:
        raise SystemExit(f"알 수 없는 MATCH_METHOD: {MATCH_METHOD}")

    m = d[d["uid"].isin(kept)]
    matched = balance_table(m, "matched")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([raw, matched], ignore_index=True).to_csv(
        OUT_DIR / "p2a_precheck_balance.csv", index=False, encoding="utf-8-sig")

    cov_cols = ["uid", "patient", "year", TARGET_ROI, "seq_pos", "seq_pos_frac",
                "patient_n_img", "patient_mean_grade", "adj_transition",
                "quality_proxy_mean", "quality_proxy_entropy"]
    d.assign(in_matched_sample=d["uid"].isin(kept).astype(int))[
        cov_cols + ["in_matched_sample"]].to_csv(
        OUT_DIR / "p2a_precheck_covariates.csv", index=False, encoding="utf-8-sig")

    m[["uid", "patient", "year", TARGET_ROI]].to_csv(
        OUT_DIR / "p2a_precheck_matched.csv", index=False, encoding="utf-8-sig")

    def counts(x):
        return {str(g): {"2024": int(((x.year == 2024) & (x[TARGET_ROI] == g)).sum()),
                         "2026": int(((x.year == 2026) & (x[TARGET_ROI] == g)).sum())}
                for g in GRADES}

    summary = {
        "features": str(FEATURES),
        "roi": TARGET_ROI,
        "grades": list(GRADES),
        "thresholds": {"smd": SMD_THRESHOLD, "auc": AUC_THRESHOLD},
        "match_method": MATCH_METHOD,
        "n_imbalanced_cells_raw": n_bad,
        "n_imbalanced_cells_matched": int(matched.imbalanced.sum()),
        "n_cells": len(raw),
        "counts_raw": counts(d),
        "counts_matched": counts(m),
        "note": ("주 분석은 정합 표본, 보조 분석은 전체 표본으로 수행한다(§3.6). "
                 "정합으로 모든 중증도 차이가 제거됐다고 주장하지 않는다."),
    }
    (OUT_DIR / "p2a_precheck_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    W = 76
    print("=" * W)
    print("precheck — Phase 2A §3.6 기존 등급 조건화 사전 점검")
    print("=" * W)
    print(f"  기준 |SMD| > {SMD_THRESHOLD} 또는 AUC > {AUC_THRESHOLD} 이면 불균형\n")
    for tag, tbl in (("정합 전", raw), ("정합 후", matched)):
        print(f"  [{tag}]  {TARGET_ROI}  2024 vs 2026")
        print(f"    {'등급':<5}{'특징':<12}{'n24':>6}{'n26':>6}{'SMD':>9}{'AUC':>8}"
              f"{'KS p':>9}{'판정':>7}")
        for _, r in tbl.iterrows():
            print(f"    {r.grade:<5}{r.feature:<12}{r.n_2024:>6}{r.n_2026:>6}"
                  f"{r.smd:>9.3f}{r.auc:>8.3f}{r.ks_p:>9.4f}"
                  f"{'  불균형' if r.imbalanced else '  OK':>7}")
        print()

    print(f"  불균형 셀  정합 전 {n_bad}/{len(raw)} → 정합 후 "
          f"{int(matched.imbalanced.sum())}/{len(matched)}   (방법 {MATCH_METHOD})")
    print(f"  {'등급':<6}{'정합 전 24/26':>16}{'정합 후 24/26':>16}")
    for g in GRADES:
        a, b = summary["counts_raw"][str(g)], summary["counts_matched"][str(g)]
        print(f"  {g:<6}{a['2024']}/{a['2026']:<14}{b['2024']}/{b['2026']:<14}")

    print("\n" + "=" * W)
    print(f"[save] {OUT_DIR}")
    for n in ("p2a_precheck_balance.csv", "p2a_precheck_covariates.csv",
              "p2a_precheck_matched.csv", "p2a_precheck_summary.json"):
        print(f"  - {n}")
    print("=" * W)


if __name__ == "__main__":
    main()
