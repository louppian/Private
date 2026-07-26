# -*- coding: utf-8 -*-
r"""
run_all — A1 검증 전체 오케스트레이션 + 최종 판정   [A1_검증실험계획 §6·§7]

순서(계획 §7): E4(양성대조·추정기검증) → E1(실데이터) → E2(raw) → E3(정합) → E5(음성대조) → 판정.
각 E 는 별도 프로세스로 실행(arm 간 GPU 메모리 격리). 마지막에 δ 보정·판정표를 낸다.

δ 보정식:  δ_corr = δ_obs - (g_r - g_f)/2 = δ_obs + Δg/2
  - g_f = 2024 학습 모델오차 ≈ g_2024,  g_r = 2026 학습 ≈ g_2026  (E1 in-domain)
  - Δg = g_2024 - g_2026  → (g_r-g_f)/2 = -Δg/2
  - E1 Δg(raw)·E2 Δg(matched) 둘로 각각 보정치를 낸다(matched 가 수축 제거본).

실행:  python run_all.py --epochs 50                 # 전체
       python run_all.py --only E1 E2                # 일부만
       python run_all.py --verdict_only             # 학습 없이 판정만(요약 json 존재 시)
"""
import argparse, json, subprocess, sys, glob, os
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "E"))
import e_common as A

HERE = Path(__file__).parent
E_DIR = HERE.parent / "E"                      # E0~E4 드라이버 위치(폴더 분리 후)
# 기존 실데이터 방향반전 run 루트 (draft §4.5) — δ_obs 추출용.
# seed 쌍(2024to2026_s* / 2026to2024_s*)이 가장 많은 run 디렉터리를 자동 선택.
BIAS_RUNS = r"D:\InhaUH_CXR\2026.05 CXRs\dorga_bias_direction_runs"


def run_step(script, extra=()):
    cmd = [sys.executable, str(E_DIR / script), *map(str, extra)]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # cp949 콘솔 크래시 방지
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(E_DIR), check=True, env=env)


def _pairs_in(run):
    pairs = []
    for fwd in glob.glob(os.path.join(run, "2024to2026_s*", "test_preds.npz")):
        s = Path(fwd).parent.name.split("_s")[-1]
        rev = os.path.join(run, f"2026to2024_s{s}", "test_preds.npz")
        if os.path.exists(rev):
            pairs.append((fwd, rev, s))
    return pairs

def _cross_seed_pairs():
    """δ_obs 소스: E1 cross(동일 early-stop 규약, checkpoint/E1/runs/dorga)."""
    pairs = _pairs_in(str(A.A1_OUT / "E1" / "runs" / "dorga"))
    return pairs, "E1 cross(early-stop 규약)"


def observed_delta():
    """cross run 에서 δ_obs(라벨성분, 보정 전) 를 seed 평균으로 ROI별 계산."""
    import numpy as np
    pairs, src = _cross_seed_pairs()
    if not pairs:
        return None
    keys = ["overall"] + list(A.ROI)
    accD = {k: [] for k in keys}; accS = {k: [] for k in keys}
    for fwd, rev, s in pairs:
        for k in keys:
            roi = None if k == "overall" else k
            d = A.decompose(fwd, rev, roi=roi, seed=int(s) if s.isdigit() else 0)
            accD[k].append(d["delta"]); accS[k].append(d["s"])
    out = {k: dict(delta=float(np.mean(accD[k])), s=float(np.mean(accS[k])),
                   n_seed=len(accD[k])) for k in keys}
    out["_source"] = src
    return out


def verdict(epochs):
    def _ld(rel):
        p = A.RESULT_OUT.joinpath(*rel.split("/"))     # summary json 은 Result 에 있음
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    e2 = _ld("E2/e2_summary.json")        # E2 raw in-domain    (A1_test)
    e3 = _ld("E3/e3_summary.json")        # E3 matched in-domain (A1_test_matched)
    e4 = _ld("E4/e4_summary.json")        # E4 양성대조 (복원곡선)
    e5 = _ld("E5/e5_summary.json")        # E5 음성대조 (누출곡선)

    dg_raw_map     = e2["A1_test"]         if e2 and "A1_test" in e2 else {}          # {key:{delta_g,..}}
    dg_matched_map = e3["A1_test_matched"] if e3 and "A1_test_matched" in e3 else {}
    dobs = observed_delta()

    dobs_roi = {k: v for k, v in dobs.items() if k != "_source"} if dobs else {}
    V = {"E2_delta_g_raw": {k: v["delta_g"] for k, v in dg_raw_map.items()},
         "E3_delta_g_matched": {k: v["delta_g"] for k, v in dg_matched_map.items()},
         "E4_recovery_slope": e4.get("recovery_slope") if e4 else None,
         "E5_leakage": [(c["train_frac"], c["delta_spurious"]) for c in e5["curve"]] if e5 else None,
         "delta_obs_source": dobs.get("_source") if dobs else None,
         "delta_obs": {k: v["delta"] for k, v in dobs_roi.items()},
         "delta_corrected": {}}

    # δ_corr[roi] = δ_obs[roi] + Δg[roi]/2  (ROI별 raw·matched 각각)
    if dobs_roi:
        for k, v in dobs_roi.items():
            row = {"delta_obs": v["delta"]}
            dgr = dg_raw_map.get(k, {}).get("delta_g")
            dgm = dg_matched_map.get(k, {}).get("delta_g")
            if dgr is not None:
                row["corr_E2raw"] = v["delta"] + dgr / 2
            if dgm is not None:
                row["corr_E3matched"] = v["delta"] + dgm / 2
            V["delta_corrected"][k] = row

    # 판정 요약 문장
    lines = ["=" * 78, "  A1 검증 최종 판정", "=" * 78]
    if e4:
        s = e4.get("recovery_slope")
        lines.append(f"[E4 양성대조] 복원 기울기 {s:+.3f} (이상 1.0) — "
                     + ("추정기 신뢰 가능" if s is not None and 0.8 <= s <= 1.2 else "추정기 편향 점검 필요"))
    if "overall" in dg_raw_map:
        t = dg_raw_map["overall"]
        lines.append(f"[E2 raw]     Δg(raw,overall) {t['delta_g']:+.4f} CI [{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] — "
                     + ("A1 위반 신호" if t["reject_A1"] else "A1 기각 못함"))
    if "overall" in dg_matched_map:
        t = dg_matched_map["overall"]
        lines.append(f"[E3 matched] Δg(matched,overall) {t['delta_g']:+.4f} CI [{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] — "
                     + ("진짜 비대칭 잔존(A1 위반)" if t["reject_A1"] else "정합 후 소멸(수축 기원)"))
    if dg_raw_map:
        lines.append("")
        lines.append(f"{'ROI':<9}{'Δg raw':>10}{'Δg matched':>12}")
        for k in ["overall"] + list(A.ROI):
            r = dg_raw_map.get(k, {}).get("delta_g"); m = dg_matched_map.get(k, {}).get("delta_g")
            lines.append(f"{k:<9}{(r if r is not None else float('nan')):>+10.3f}"
                         f"{(m if m is not None else float('nan')):>+12.3f}")
    if V["delta_corrected"]:
        lines.append("")
        lines.append(f"{'ROI':<9}{'δ_obs':>10}{'δ_corr(E2raw)':>16}{'δ_corr(E3matched)':>19}")
        for k, row in V["delta_corrected"].items():
            lines.append(f"{k:<9}{row['delta_obs']:>+10.3f}"
                         f"{row.get('corr_E2raw', float('nan')):>+16.3f}"
                         f"{row.get('corr_E3matched', float('nan')):>+19.3f}")
        lines.append("")
        lines.append("→ RB·LT 의 δ_corr 이 여전히 양(+)으로 크면 라벨 드리프트 결론 유지,")
        lines.append("  0 근처로 붕괴하면 관측 δ 는 모델 방향비대칭의 산물이었음.")
    txt = "\n".join(lines)
    A.save_json(V, A.RESULT_OUT / "A1_verdict.json")           # 판정 json 은 Result (git 추적)
    (A.RESULT_OUT / "A1_verdict.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print("\nsaved:", A.RESULT_OUT / "A1_verdict.json", "/", A.RESULT_OUT / "A1_verdict.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["E1", "E2", "E3", "E4", "E5"],
                    choices=["E1", "E2", "E3", "E4", "E5"])
    ap.add_argument("--verdict_only", action="store_true")
    args = ap.parse_args()

    if not args.verdict_only:
        order = [s for s in ["E1", "E2", "E3", "E4", "E5"] if s in args.only]
        script = {"E1": "e1_cross.py", "E2": "e2_indomain_raw.py",
                  "E3": "e3_indomain_matched.py", "E4": "e4_positive_control.py",
                  "E5": "e5_negative_control.py"}
        for step in order:
            run_step(script[step])                     # epochs 고정(50) → 인자 없음

    verdict(A.B.EPOCHS)


if __name__ == "__main__":
    main()
