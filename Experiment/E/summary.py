# -*- coding: utf-8 -*-
r"""
E 결과 집계 — E1~E5 단일 진입점 (舊 A/run_all.py 의 판정 로직 이관).

- E1 cross : results.json 10개(fwd/rev × split1~5) → 방향별 test bias 평균±sd,
             ROI별 δ_obs=(rev−fwd)/2 · γ=(rev+fwd)/2 · 방향반전(flip) 판정.
- E2 (matched) : e2_summary.json → Δg CSV (matched). 부록B(raw)는 appb_summary.json.
- 최종 판정 : δ_corr = δ_obs + Δγ_matched/2 (환자 부트스트랩 95% CI) → Result/A1_verdict.json.
- E3/E4    : 복원 기울기(양성대조)·누출(음성대조) 요약 출력.

check_value_{l,e,a}.py 는 이 산출물을 draft md 값과 '비교만' 한다(계산 안 함).

산출:
  Result/E1/e1_summary.csv   fwd/rev/δ/γ/flip
  Result/E2/e2_summary.csv           Δg_matched (Experiment 2)
  Result/AppendixB/appb_summary.csv  Δg_raw (부록 B)
  Result/A1_verdict.json     δ_obs·γ·Δγ·δ_corr+CI (최종 판정)

실행:
  python Experiment/E/summary.py
  python Experiment/E/summary.py --runs /path/to/Result/E1/dorga
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):              # Windows cp949 콘솔 유니코드(— 등) 크래시 방지
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

_HERE = Path(__file__).resolve().parent          # Experiment/E
_REPO = _HERE.parents[1]                          # Private repo 루트
RUNS_DIR   = _REPO / "Result" / "E1" / "dorga"       # 입력: {24to26,26to24}_split{k}/results.json
CKPT_E1    = _REPO / "checkpoint" / "E1" / "dorga"   # 입력: npz (환자 부트스트랩 CI 용)
RESULT_E1  = _REPO / "Result" / "E1"
RESULT_MATCHED = _REPO / "Result" / "E2"           # Experiment 2 (matched in-domain, Δγ)
RESULT_POS     = _REPO / "Result" / "E3"           # Experiment 3 (양성대조, 복원곡선)
RESULT_NEG     = _REPO / "Result" / "E4"           # Experiment 4 (음성대조, 누출)
RESULT_APPB    = _REPO / "Result" / "AppendixB"    # 부록 B (raw in-domain, Δg_raw)
RESULT_ROOT = _REPO / "Result"
ROI = ["RT", "LT", "RB", "LB"]
NBOOT = 5000


def _dir(mode):
    return "fwd" if str(mode).startswith("24to26") else "rev"


def load_all(runs_dir: Path):
    recs = []
    for p in sorted(runs_dir.glob("*/results.json")):     # {tag}_split{k}/results.json
        try:
            recs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return recs


def ms(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def fmt(m, s):
    return f"{m:+.3f}±{s:.3f}" if not np.isnan(m) else "     -"


# ═══════════════ E2/E3 in-domain Δg CSV export ═══════════════
def _export_indomain(summary_json, a1_key, dg_col, reject_col, out_dir, out_name):
    if not summary_json.exists():
        return
    d = json.loads(summary_json.read_text(encoding="utf-8"))
    a1 = d.get(a1_key, {})
    if not a1:
        return
    rows = []
    for roi in ["overall"] + ROI:
        t = a1.get(roi, {})
        ci = t.get("ci", [None, None]) or [None, None]
        rows.append({"roi": roi,
                     dg_col: round(t["delta_g"], 4) if "delta_g" in t else "",
                     "ci_lo": round(ci[0], 4) if ci[0] is not None else "",
                     "ci_hi": round(ci[1], 4) if ci[1] is not None else "",
                     reject_col: int(t["reject_A1"]) if "reject_A1" in t else ""})
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / out_name
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["roi", dg_col, "ci_lo", "ci_hi", reject_col])
        w.writeheader(); w.writerows(rows)
    print(f"[save] {p}")


def export_matched():   # Experiment 2 (matched) → Result/E2/e2_summary.csv (Δg_matched)
    _export_indomain(RESULT_MATCHED / "e2_summary.json", "A1_test_matched",
                     "delta_g_matched", "reject_A1_matched", RESULT_MATCHED, "e2_summary.csv")


def export_appb():      # 부록 B (raw) → Result/AppendixB/appb_summary.csv (Δg_raw)
    _export_indomain(RESULT_APPB / "appb_summary.json", "A1_test",
                     "delta_g_raw", "reject_A1_raw", RESULT_APPB, "appb_summary.csv")


# ═══════════════ 환자 bias 부트스트랩 (E1 npz, split 전부 pool) ═══════════════
def _pooled_patient_bias(mode, roi):
    """checkpoint/E1/dorga/{mode}_split*/test_preds.npz 를 전부 pool → 환자별 bias 벡터."""
    files = sorted(CKPT_E1.glob(f"{mode}_split*/test_preds.npz"))
    if not files:
        return None
    Ps, Ys, PT = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        Ps.append(d["preds"]); Ys.append(d["labels"])
        PT.append(np.asarray(d["patients"]).astype(str))
    P, Y, pt = np.vstack(Ps), np.vstack(Ys), np.concatenate(PT)
    if roi is None:
        e = (P - Y).astype(float).mean(1)
    else:
        j = ROI.index(roi); e = (P[:, j] - Y[:, j]).astype(float)
    u = np.unique(pt)
    return np.array([e[pt == up].mean() for up in u])


def _boot_mean(vec, n=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    return vec[rng.integers(0, len(vec), (n, len(vec)))].mean(1)


def _boot_mean_joint(mat, seed, n=NBOOT):
    """행(환자)을 통째로 재표집. 열(ROI/overall) 간 공분산이 보존된다.

    ROI 를 하나씩 따로 부트스트랩하면 주변 CI 는 맞지만 ROI 간 공분산이 사라져
    ROI 대비(RB − LT 등)를 낼 수 없다. 여기서 인덱스를 한 번만 뽑아 전 열에 공유한다.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mat.shape[0], (n, mat.shape[0]))
    return mat[idx].mean(axis=1)


# ROI 대비 — RB 가 사전 음성 참조보다 큰지. LT 는 Phase 1 에서 후보 가능성이 있어
# 사전 음성 참조에서 제외한다(draft §5.3).
CONTRASTS = {
    "RB_minus_LT": ["LT"],
    "RB_minus_mean_RT_LT_LB": ["RT", "LT", "LB"],
    "RB_minus_mean_RT_LB": ["RT", "LB"],
}


# ═══════════════ 최종 판정 δ_corr = δ_obs + Δγ_matched/2 ═══════════════
def build_verdict(dobs_map=None):
    """δ_obs 점추정은 E1 split 평균(dobs_map, results.json 유래 = git 재현) **단일 소스**를 쓴다.
    E1 표와 δ_corr 이 같은 δ_obs 를 공유하므로 두 값이 갈리지 않는다.
    Δγ 점추정·CI 는 e3_summary.json(git). δ_obs·δ_corr 의 CI 는 npz 환자 부트스트랩 폭을
    점추정에 중심맞춰 산출(npz 없으면 CI 생략, 점추정만). → Result/A1_verdict.json."""
    def _ld(p):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    matched = _ld(RESULT_MATCHED / "e2_summary.json")   # Δγ (Experiment 2, matched)
    raw     = _ld(RESULT_APPB / "appb_summary.json")     # Δγ_raw (부록 B, 정합 전)
    pos     = _ld(RESULT_POS / "e3_summary.json")        # 복원곡선 (Experiment 3, 양성대조)
    neg     = _ld(RESULT_NEG / "e4_summary.json")        # 누출 (Experiment 4, 음성대조)

    def _reci(boot, point):                          # 부트스트랩 폭을 점추정 중심에 맞춤
        lo, hi = np.percentile(boot, [2.5, 97.5]); m = float(boot.mean())
        return [float(point + (lo - m)), float(point + (hi - m))]

    keys = [k for k in (["overall"] + ROI) if dobs_map and k in dobs_map]
    if not keys:
        keys = []

    # ── 열(키) 단위로 환자 벡터를 모아 한 번에 부트스트랩한다 ──
    def _stack(getter):
        cols = []
        for key in keys:
            v = getter(key)
            if v is None:
                return None
            cols.append(np.asarray(v, dtype=float))
        if len({len(c) for c in cols}) != 1:            # 키마다 환자 수가 다르면 결합 불가
            return None
        return np.column_stack(cols)

    fwd = _stack(lambda k: _pooled_patient_bias("24to26", None if k == "overall" else k))
    rev = _stack(lambda k: _pooled_patient_bias("26to24", None if k == "overall" else k))
    have_npz = fwd is not None and rev is not None

    def _indomain(src, root, year):
        if not (src and root in src):
            return None
        return _stack(lambda k: src[root][year][k]["vec"] if k in src[root][year] else None)

    m24, m26 = _indomain(matched, "matched", "2024"), _indomain(matched, "matched", "2026")
    r24, r26 = _indomain(raw, "raw", "2024"), _indomain(raw, "raw", "2026")

    dboot = None
    if have_npz:                                        # δ_obs 구름 (키 간 공분산 보존)
        dboot = (_boot_mean_joint(rev, seed=2) - _boot_mean_joint(fwd, seed=1)) / 2
    corr_boot = corr_boot_r = None
    if dboot is not None and m24 is not None and m26 is not None:
        corr_boot = dboot + (_boot_mean_joint(m24, seed=3) - _boot_mean_joint(m26, seed=4)) / 2
    if dboot is not None and r24 is not None and r26 is not None:
        corr_boot_r = dboot + (_boot_mean_joint(r24, seed=5) - _boot_mean_joint(r26, seed=6)) / 2

    dc = {}
    for i, key in enumerate(keys):
        dobs = float(dobs_map[key]["delta_obs"]); gamma = float(dobs_map[key]["gamma"])
        row = {"delta_obs": dobs, "gamma": gamma}
        if dboot is not None:
            row["delta_obs_ci"] = _reci(dboot[:, i], dobs)
        if m24 is not None and m26 is not None:
            dg = float(m24[:, i].mean() - m26[:, i].mean())
            row["delta_g_matched"] = dg
            row["delta_corr"] = dobs + dg / 2
            if corr_boot is not None:
                row["ci"] = _reci(corr_boot[:, i], row["delta_corr"])
        if r24 is not None and r26 is not None:          # 부록 B: 정합 전 δ_corr_raw
            dgr = float(r24[:, i].mean() - r26[:, i].mean())
            row["delta_g_raw"] = dgr
            row["delta_corr_raw"] = dobs + dgr / 2
            if corr_boot_r is not None:
                row["ci_raw"] = _reci(corr_boot_r[:, i], row["delta_corr_raw"])
        dc[key] = row

    # ── ROI 대비 (탐색적, 다중성 미보정) ──
    contrasts, corr_mat = None, None
    if corr_boot is not None and all(r in keys for r in ROI):
        col = {k: i for i, k in enumerate(keys)}
        # 점추정 중심으로 맞춘 구름에서 대비를 뽑는다(주변 CI 관례와 동일).
        pt = np.array([dc[k].get("delta_corr", np.nan) for k in keys], dtype=float)
        cloud = corr_boot - corr_boot.mean(axis=0) + pt
        contrasts = {}
        for name, others in CONTRASTS.items():
            v = cloud[:, col["RB"]] - sum(cloud[:, col[o]] for o in others) / float(len(others))
            lo, hi = np.percentile(v, [2.5, 97.5])
            contrasts[name] = {
                "estimate": float(v.mean()),
                "ci_95_two_sided": [float(lo), float(hi)],
                "p_one_sided_gt_0": float(np.mean(v <= 0.0)),
            }
        sub = cloud[:, [col[r] for r in ROI]]
        corr_mat = {"roi_order": list(ROI), "matrix": np.corrcoef(sub, rowvar=False).tolist()}

    V = {"delta_corrected": dc,
         "roi_contrasts": contrasts,
         "roi_contrast_note": ("δ_corr 의 환자 단위 joint bootstrap 구름에서 산출. "
                               "탐색적 분석이며 다중성을 보정하지 않았다. CI 는 양측 95% "
                               "percentile, p 는 단측 bootstrap tail 비율이다."),
         "roi_correlation": corr_mat,
         "E3_recovery_slope": (pos.get("recovery_slope") if pos else None),
         "E4_leakage": ([(c["train_frac"], c["delta_spurious"]) for c in neg["curve"]] if neg else None)}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "A1_verdict.json").write_text(
        json.dumps(V, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print("  최종 판정  δ_corr = δ_obs + Δγ_matched/2   (환자 부트스트랩 95% CI)")
    print("=" * 78)
    if not dc:
        print("  [SKIP] E1 npz 없음 (checkpoint/E1/dorga) — e1_cross.py 실행 후")
    else:
        print(f"  {'ROI':<8}{'δ_obs':>9}{'γ':>9}{'Δγ_m':>9}{'δ_corr':>9}{'95% CI':>22}{'CI0배제':>9}")
        for key in keys:
            if key not in dc:
                continue
            row = dc[key]
            dcv, ci = row.get("delta_corr"), row.get("ci")
            ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "-"
            excl = ("예" if (ci and ci[0] * ci[1] > 0) else "아니오") if ci else "-"
            print(f"  {key:<8}{row['delta_obs']:>+9.3f}{row['gamma']:>+9.3f}"
                  f"{row.get('delta_g_matched', float('nan')):>+9.3f}"
                  f"{(dcv if dcv is not None else float('nan')):>+9.3f}{ci_s:>22}{excl:>9}")
        if any("delta_corr_raw" in dc[k] for k in dc):     # 부록 B: 정합 전 δ_corr_raw (대조)
            print(f"\n  [부록 B] 정합 전 δ_corr_raw = δ_obs + Δγ_raw/2")
            for key in keys:
                row = dc.get(key, {})
                if "delta_corr_raw" not in row:
                    continue
                cir = row.get("ci_raw")
                cir_s = f"[{cir[0]:+.3f},{cir[1]:+.3f}]" if cir else "-"
                print(f"    {key:<8}Δγ_raw {row['delta_g_raw']:>+.3f}  "
                      f"δ_corr_raw {row['delta_corr_raw']:>+.3f}  {cir_s}")
        if contrasts:
            print(f"\n  [ROI 대비] δ_corr joint bootstrap · 탐색적 · 다중성 미보정")
            print(f"    {'대비':<24}{'추정':>9}{'양측 95% CI':>22}{'단측 p':>9}")
            for name, c in contrasts.items():
                lo, hi = c["ci_95_two_sided"]
                print(f"    {name:<24}{c['estimate']:>+9.4f}"
                      f"{f'[{lo:+.3f},{hi:+.3f}]':>22}{c['p_one_sided_gt_0']:>9.4f}")
        if corr_mat:
            print(f"\n  [ROI 상관] δ_corr bootstrap 구름")
            print("    " + " " * 6 + "".join(f"{r:>8}" for r in corr_mat["roi_order"]))
            for i, r in enumerate(corr_mat["roi_order"]):
                print(f"    {r:<6}" + "".join(f"{v:>8.3f}" for v in corr_mat["matrix"][i]))
        if pos and pos.get("recovery_slope") is not None:
            print(f"\n  [E3 양성대조] 복원 기울기 {pos['recovery_slope']:+.3f} (이상 1.0)")
        if neg and neg.get("curve"):
            leak = ", ".join(f"frac{c['train_frac']}={c['delta_spurious']:+.3f}" for c in neg["curve"])
            print(f"  [E4 음성대조] 누출: {leak}")
    print(f"[save] {RESULT_ROOT / 'A1_verdict.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS_DIR))
    args = ap.parse_args()
    runs_dir = Path(args.runs)

    export_matched()                    # Experiment 2 matched Δg → Result/E2/e2_summary.csv
    export_appb()                       # 부록 B raw Δg           → Result/AppendixB/appb_summary.csv

    recs = load_all(runs_dir)
    if not recs:
        print(f"[중단] cross results.json 없음: {runs_dir}")
        build_verdict()                 # E3/E4/E5 CSV 는 위에서 저장됨; 판정은 npz 있으면 시도
        return
    print(f"[load] results.json {len(recs)}개  ({runs_dir})")

    G = {"fwd": [], "rev": []}
    for r in recs:
        G[_dir(r["mode"])].append(r)
    fwd, rev = G["fwd"], G["rev"]

    csv_rows = []
    head = f"[dorga]  fwd(2024→2026) vs rev(2026→2024)  | splits fwd={len(fwd)} rev={len(rev)}"
    print("\n" + "=" * 78); print(head); print("=" * 78)

    for tag, rr in (("fwd", fwd), ("rev", rev)):
        if not rr:
            continue
        mae = ms([r["mae"] for r in rr]); acc = ms([r["acc"] for r in rr]); bia = ms([r["bias"] for r in rr])
        line = f"  [{tag}] test  MAE {fmt(*mae).replace('+', '')}  " \
               f"ACC {fmt(*acc).replace('+', '')}  overall bias {fmt(*bia)}"
        print(line)

    print(f"\n  {'ROI':<6}{'fwd bias':>16}{'rev bias':>16}{'δ_obs':>10}{'γ':>10}{'flip':>7}")
    n_flip = 0
    for roi in ["overall"] + ROI:
        if roi == "overall":
            fb = [r["bias"] for r in fwd]; rb = [r["bias"] for r in rev]
        else:
            fb = [r["per_roi"][roi]["bias"] for r in fwd]
            rb = [r["per_roi"][roi]["bias"] for r in rev]
        fm, fs = ms(fb); rm, rs = ms(rb)
        flip = (not np.isnan(fm) and not np.isnan(rm) and fm < 0 < rm)
        if roi != "overall" and flip:
            n_flip += 1
        d_obs = (rm - fm) / 2 if not (np.isnan(fm) or np.isnan(rm)) else float("nan")
        gam = (rm + fm) / 2 if not (np.isnan(fm) or np.isnan(rm)) else float("nan")
        mark = "YES" if flip else ("no" if not np.isnan(fm) else "-")
        print(f"  {roi:<6}{fmt(fm, fs):>16}{fmt(rm, rs):>16}{d_obs:>+10.3f}{gam:>+10.3f}{mark:>7}")
        csv_rows.append(dict(model="dorga", roi=roi,
                             fwd_bias_mean=round(fm, 4), fwd_bias_sd=round(fs, 4),
                             rev_bias_mean=round(rm, 4), rev_bias_sd=round(rs, 4),
                             delta_obs=round(d_obs, 4), gamma=round(gam, 4),
                             flip_Hdata=int(flip)))

    print(f"  → dorga: 4개 영역 중 {n_flip}개에서 방향반전(fwd<0<rev)")

    RESULT_E1.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_E1 / "e1_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "roi", "fwd_bias_mean", "fwd_bias_sd",
                                          "rev_bias_mean", "rev_bias_sd",
                                          "delta_obs", "gamma", "flip_Hdata"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"[save] {csv_path}")

    dobs_map = {row["roi"]: row for row in csv_rows}   # E1 표와 동일한 split 평균 δ_obs 단일 소스
    build_verdict(dobs_map)             # δ_corr + CI → Result/A1_verdict.json


if __name__ == "__main__":
    main()
