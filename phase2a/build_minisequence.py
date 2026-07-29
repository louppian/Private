# -*- coding: utf-8 -*-
r"""build_minisequence.py — Phase 2A §3.2 대상군 선정 + §3.4 mini-sequence 구성.

서버의 CXR/Merged 만 읽는다. 영상 파일은 만들지도 복사하지도 않는다.

  §3.2 대상군
    표적군          2024 RB 기존 등급 3·4      203 ROI 전부
    하위 경계 대조군  2024 RB 기존 등급 2        환자·등급 층화 (등급 3은 표적군과 공유)
    원거리 음성 대조군 2024 RB 기존 등급 0·1     환자·등급 층화
    연도 대조군      2026 RB 기존 등급 3·4      환자 층화
    hidden duplicate 전체 판독 세트의 10%       표적군·대조군 층화

  §3.4 mini-sequence
    환자 전체 시퀀스를 보여주지 않는다. 환자 내부는 시간순, 환자 간은 무작위.
    3·4 가 5장 이하  직전 1장 + 구간 전부 + 직후 1장 (환자당 3~7장)
    3·4 가 6장 이상  첫·중간·마지막 + 3↔4 전이 직전·직후 + 구간 밖 anchor 각 1장,
                    중복 제거 후 환자당 최대 6~7장
    대조군          환자당 연속 3~5장

산출 (Result/P2A/)
  p2a_reading_sheet.csv    판독자용. 연도·기존등급·대상군·중복여부 전부 가림
  p2a_admin_key.csv        분석용. 위 정보 + §3.4 기록 항목
  p2a_image_manifest.csv   case_id → Merged 원본 경로 (판독 폴더 만들 때 쓰는 이름표)
  p2a_summary.json         대상군별 영상·환자 수

CLI 대신 아래 상수를 직접 수정한다.
"""

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

REPO = Path(__file__).resolve().parent.parent
ROI = ("RT", "LT", "RB", "LB")

# ═══════════ 경로 — 서버 기준(core.py 와 동일). 여기만 맞게 수정 ═══════════
BASE = "/shared/home/mai/JeongGeon/Private"
MERGED = Path(f"{BASE}/CXR/Merged")                      # labels.csv image_path 기준 폴더
IMG_DIR = Path(f"{BASE}/CXR/Merged/images_normalize")    # 정렬 완료 영상 <uid>.png
LABELS_CSV = f"{BASE}/CXR/Merged/labels.csv"
OUT_DIR = REPO / "Result" / "P2A"

UID_COL, PATIENT_COL, YEAR_COL, IMAGE_COL = "uid", "patient_id", "year", "image_path"
TARGET_ROI = "RB"                  # 대상군 선정 기준 ROI. 판독은 4 ROI 전부(§3.3)
TIME_COLS = ["study_date", "date", "timepoint", "series_index", "uid"]

# ── §3.4 ──
SHORT_RUN_MAX = 5                  # 3·4 영상이 이 수 이하면 "구간 전부 + 앞뒤 anchor"
MAX_MINISEQ = 7                    # 6장 이상 분기의 환자당 상한 (plan 6~7)
CONTROL_SEQ_LEN = 5                # 대조군 환자당 연속 영상 수 (plan 3~5)

# §3.2 대조군은 "환자·등급 층화" 선정이다. 조건 맞는 환자를 전부 넣으면 2024 는
# 사실상 전원(68명)이 되어 표적군 45명보다 대조군이 커진다.
CONTROL_MAX_PATIENTS = None        # None 이면 표적군 환자 수와 동수
STRATA_BINS = 3                    # 시퀀스 길이·환자 평균 등급 각각의 분위 수

# hidden duplicate 비율 기준: 원본 판독 대상(중복 제외) 대비 비율로 고정한다.
# "전체 판독 세트의 10%" 를 최종 행 수 기준으로 읽으면 quota 와 출력이 어긋난다.
# §3.2 는 표적군을 "203 ROI 전부" 로, §3.4 는 6장 이상 환자에서 대표영상만 뽑도록
# 규정한다. 두 조항이 충돌하므로 어느 쪽을 따를지 여기서 정한다.
#   False  §3.4 우선 — 6장 이상 환자는 대표영상만 (표적군이 203 미만이 된다)
#   True   §3.2 우선 — 2024 RB 3·4 를 전부 넣고 §3.4 상한을 표적군에는 적용하지 않는다
TARGET_FULL_ENUMERATION = False

# ── §3.2 hidden duplicate ──
# 낱장을 같은 시퀀스 뒤에 붙이면 판독자가 즉시 알아채므로 mini-sequence 단위로
# 재제시하고 새 blind 환자 ID 를 준다. 새 영상 파일을 만들지 않고 같은 원본을
# 다른 case_id 로 한 번 더 가리킬 뿐이다.
DUPLICATE_FRAC = 0.10
MIN_DUP_GAP = 5                    # 원본 시퀀스와 중복 시퀀스 사이 최소 시퀀스 간격

BLIND_IMAGE_EXT = ".png"
SEED = 20260729
SALT = "phase2a"


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def infer_year(row, has_year_col):
    if has_year_col and str(row.get(YEAR_COL, "")).strip():
        return str(row[YEAR_COL]).strip()
    return {"24": "2024", "26": "2026"}.get(str(row.get(PATIENT_COL, ""))[:2], "")


def int_grade(x):
    try: return int(float(str(x).strip()))
    except Exception: return None


def sort_key(row, time_cols):
    return tuple(str(row.get(c, "")) for c in time_cols) + (str(row.get(UID_COL, "")),)


def hash_id(value, n=10):
    return hashlib.sha256((SALT + "::" + value).encode("utf-8")).hexdigest()[:n].upper()


def source_path(row, has_image_col):
    """Merged 안의 실제 영상 경로. image_path 는 MERGED 기준 상대경로다(core.py)."""
    rel = str(row.get(IMAGE_COL, "")).strip() if has_image_col else ""
    if rel:
        p = Path(rel)
        return str(p if p.is_absolute() else MERGED / rel)
    return str(IMG_DIR / (row["_uid"] + BLIND_IMAGE_EXT))


# ────────────────────────────── §3.4 mini-sequence ──────────────────────────────

def upper_minisequence(grades):
    """기존 등급 3·4 구간의 mini-sequence. {index: 선정사유} 반환."""
    n = len(grades)
    target = [i for i in range(n) if grades[i] in (3, 4)]
    if not target:
        return {}

    anchors = [i for i in (target[0] - 1, target[-1] + 1) if 0 <= i < n]

    # 5장 이하: 직전 1장 + 구간 전부 + 직후 1장
    if len(target) <= SHORT_RUN_MAX or TARGET_FULL_ENUMERATION:
        out = {i: "target" for i in target}
        out.update({i: "anchor" for i in anchors if i not in out})
        return dict(sorted(out.items()))

    # 6장 이상: 대표영상만. plan 이 나열한 순서를 우선순위로 채운다.
    # 무작위로 버리면 사전 지정한 전이 영상이 빠진다.
    tiers = (
        ("span", [target[0], target[len(target) // 2], target[-1]]),
        ("transition", [i for a, b in zip(target, target[1:]) if grades[a] != grades[b]
                        for i in (a, b)]),
        ("anchor", anchors),
    )
    out = {}
    for why, tier in tiers:
        for i in tier:
            if len(out) >= MAX_MINISEQ:
                break
            out.setdefault(i, why)
    return dict(sorted(out.items()))


def control_minisequence(grades, eligible, rng):
    """대조군 연속 3~5장. eligible 등급을 가장 많이 담는 창 중 무작위 하나.

    '조건을 만족하는 가장 긴 창' 으로 고르면 사실상 항상 시퀀스 앞쪽이 뽑힌다.
    """
    if not eligible:
        return set()
    n = len(grades)
    best, cands = 0, []
    for s in range(max(1, n - CONTROL_SEQ_LEN + 1)):
        win = list(range(s, min(n, s + CONTROL_SEQ_LEN)))
        hit = sum(i in eligible for i in win)
        if hit > best:
            best, cands = hit, [win]
        elif hit == best and hit > 0:
            cands.append(win)
    return set(rng.choice(cands)) if cands else set()


# ────────────────────────────── §3.2 대상군 ──────────────────────────────

def mark(store, row, flags, reason):
    rec = store.setdefault(row["_uid"], dict(row, _reasons=set()))
    rec["_reasons"].add(reason)
    for k, v in flags.items():
        rec[k] = max(int(rec.get(k, 0)), int(v))


def profile(seq):
    """환자 프로파일 — (시퀀스 길이, 환자 평균 등급)."""
    g = [r["_grade"] for r in seq]
    return len(seq), sum(g) / len(g)


def quantile_edges(vals, k):
    """분위 경계. 동률이 많으면 실제 층 수가 k 보다 줄어든다."""
    if not vals:
        return []
    s = sorted(vals)
    return [s[min(int(i / k * len(s)), len(s) - 1)] for i in range(1, k)]


def to_bin(v, edges):
    return sum(1 for e in edges if v > e)


def choose_control_patients(by_patient, target_keys, eligible, cap, rng):
    """§3.2 환자·등급 층화 — 표적군의 (시퀀스 길이 × 평균 등급) 층 분포에 맞춰 뽑는다."""
    if not eligible or cap <= 0:
        return set()
    prof = {k: profile(by_patient[k]) for k in set(target_keys) | set(eligible)}
    len_edges = quantile_edges([prof[k][0] for k in target_keys], STRATA_BINS)
    gr_edges = quantile_edges([prof[k][1] for k in target_keys], STRATA_BINS)

    def stratum(k):
        n, m = prof[k]
        return to_bin(n, len_edges), to_bin(m, gr_edges)

    want = Counter(stratum(k) for k in target_keys)
    n_target = max(sum(want.values()), 1)

    pool = defaultdict(list)
    for k in eligible:
        pool[stratum(k)].append(k)
    for s in pool:
        pool[s].sort()
        rng.shuffle(pool[s])

    chosen = set()
    for s, c in sorted(want.items()):
        take = int(round(cap * c / n_target))
        chosen.update(pool[s][:take])
        pool[s] = pool[s][take:]

    # 층이 비어 못 채운 몫은 남은 환자에서 무작위로 채운다.
    rest = [k for s in sorted(pool) for k in pool[s]]
    rng.shuffle(rest)
    for k in rest:
        if len(chosen) >= cap:
            break
        chosen.add(k)
    return chosen


def select_cases(rows, rng):
    by_patient = defaultdict(list)
    for r in rows:
        if r["_year"] in ("2024", "2026") and r["_grade"] is not None:
            by_patient[(r["_year"], r["_patient"])].append(r)
    for k in by_patient:
        by_patient[k].sort(key=lambda r: sort_key(r, r["_time_cols"]))

    def has(key, grades):
        return any(r["_grade"] in grades for r in by_patient[key])

    keys24 = sorted(k for k in by_patient if k[0] == "2024")
    keys26 = sorted(k for k in by_patient if k[0] == "2026")

    target_keys = [k for k in keys24 if has(k, (3, 4))]
    year_keys = [k for k in keys26 if has(k, (3, 4))]
    cap = CONTROL_MAX_PATIENTS if CONTROL_MAX_PATIENTS else len(target_keys)
    lower_keys = choose_control_patients(
        by_patient, target_keys, [k for k in keys24 if has(k, (2,))], cap, rng)
    distant_keys = choose_control_patients(
        by_patient, target_keys, [k for k in keys24 if has(k, (0, 1))], cap, rng)

    store = {}
    # 표적군 — 2024 기존 등급 3·4 (등급 3 은 하위 경계 대조군과 공유)
    for key in target_keys:
        seq = by_patient[key]
        grades = [r["_grade"] for r in seq]
        for i, why in upper_minisequence(grades).items():
            mark(store, seq[i], {"target_2024_rb_upper": int(grades[i] in (3, 4)),
                                 "control_2024_rb_lower": int(grades[i] == 3)}, why)

    # 연도 대조군 — 2026 기존 등급 3·4
    for key in year_keys:
        seq = by_patient[key]
        grades = [r["_grade"] for r in seq]
        for i, why in upper_minisequence(grades).items():
            mark(store, seq[i], {"control_2026_rb_upper": int(grades[i] in (3, 4))}, why)

    # 하위 경계 대조군 — 2024 기존 등급 2
    for key in sorted(lower_keys):
        seq = by_patient[key]
        grades = [r["_grade"] for r in seq]
        elig = {i for i in range(len(seq)) if grades[i] == 2}
        for i in control_minisequence(grades, elig, rng):
            mark(store, seq[i], {"control_2024_rb_lower": int(grades[i] in (2, 3))}, "lower_ctl")

    # 원거리 음성 대조군 — 2024 기존 등급 0·1
    for key in sorted(distant_keys):
        seq = by_patient[key]
        grades = [r["_grade"] for r in seq]
        elig = {i for i in range(len(seq)) if grades[i] in (0, 1)}
        for i in control_minisequence(grades, elig, rng):
            mark(store, seq[i], {"control_2024_rb_distant": int(grades[i] in (0, 1))}, "distant_ctl")

    for rec in store.values():
        rec["_patient_total"] = len(by_patient[(rec["_year"], rec["_patient"])])
    return store, {"target": len(target_keys), "year_ctl": len(year_keys),
                   "lower_ctl": len(lower_keys), "distant_ctl": len(distant_keys),
                   "control_cap": cap}


def primary_group(members):
    for flag in ("target_2024_rb_upper", "control_2026_rb_upper",
                 "control_2024_rb_lower", "control_2024_rb_distant"):
        if any(int(m.get(flag, 0)) for m in members):
            return flag
    return "context_only"


def build_sequences(store, rng):
    groups = defaultdict(list)
    for rec in store.values():
        groups[(rec["_year"], rec["_patient"])].append(rec)
    for k in groups:
        groups[k].sort(key=lambda r: sort_key(r, r["_time_cols"]))

    seqs = []
    for key, members in sorted(groups.items()):
        blind = "P" + hash_id(f"{key[0]}|{key[1]}")
        seqs.append({"sid": "S" + blind, "blind_patient": blind, "src_key": key,
                     "members": members, "dup_of": ""})

    # 대상군 구성비를 유지한 채 원본 판독 대상의 10% 만큼 시퀀스를 재제시한다(§3.2 층화).
    # mini-sequence 단위라 정확히 10% 를 못 맞추므로 "초과하지 않음" 을 우선한다.
    total = sum(len(s["members"]) for s in seqs)
    quota = round(total * DUPLICATE_FRAC)
    strata = defaultdict(list)
    for s in seqs:
        strata[primary_group(s["members"])].append(s)

    dups, used = [], 0
    for name in sorted(strata):
        pool = strata[name][:]
        rng.shuffle(pool)
        share = quota * sum(len(s["members"]) for s in strata[name]) / max(total, 1)
        taken = 0
        for s in pool:
            n = len(s["members"])
            if taken + n > share or used + n > quota:
                continue
            blind = "P" + hash_id(f"dup|{s['src_key'][0]}|{s['src_key'][1]}")
            dups.append({"sid": "S" + blind, "blind_patient": blind, "src_key": s["src_key"],
                         "members": s["members"], "dup_of": s["sid"]})
            taken += n
            used += n
    return seqs, dups


def order_sequences(seqs, dups, rng):
    """환자 간 순서 무작위화(§3.4). 중복 시퀀스는 원본에서 MIN_DUP_GAP 이상 떨어뜨린다."""
    order = seqs + dups
    rng.shuffle(order)
    bad = []
    for _ in range(200):
        pos = {s["sid"]: i for i, s in enumerate(order)}
        bad = [i for i, s in enumerate(order)
               if s["dup_of"] and (i - pos[s["dup_of"]]) < MIN_DUP_GAP]
        if not bad:
            break
        s = order.pop(bad[0])
        lo = min(len(order), pos[s["dup_of"]] + MIN_DUP_GAP)
        order.insert(rng.randint(lo, len(order)), s)
    if bad:
        print(f"[warn] 중복 시퀀스 {len(bad)}개가 MIN_DUP_GAP={MIN_DUP_GAP} 을 못 지켰다")
    return order


def emit(order, has_image_col):
    """판독시트·admin key·manifest 생성.

    case_id 는 셔플된 최종 판독 순서로 매긴다. 선정 순서(연도·환자 정렬)로 매기면
    번호 자체가 연도를 노출한다(§3.1 조건 1).
    """
    reading, admin, manifest = [], [], []
    case_no = 0
    for s in order:
        seq_reasons = {x for m in s["members"] for x in m["_reasons"]}
        for within, m in enumerate(s["members"], 1):
            case_no += 1
            cid = f"C{case_no:05d}"
            blind_img = cid + BLIND_IMAGE_EXT
            src = source_path(m, has_image_col)

            reading.append({
                "case_id": cid,
                "blind_patient_id": s["blind_patient"],
                "sequence_id": s["sid"],
                "within_sequence_order": within,
                "global_order": case_no,
                "image_file": blind_img,
                "read_RT": "", "read_LT": "", "read_RB": "", "read_LB": "",
                "non_evaluable": "", "notes": "",
            })
            manifest.append({"case_id": cid, "image_file": blind_img, "source_path": src,
                             "duplicate_flag": int(bool(s["dup_of"]))})
            admin.append({
                "case_id": cid,
                "uid": m["_uid"],
                "patient_id": m["_patient"],
                "year": m["_year"],
                "blind_patient_id": s["blind_patient"],
                "sequence_id": s["sid"],
                "within_sequence_order": within,
                "global_order": case_no,
                "source_path": src,
                **{f"orig_{r}": m.get(r, "") for r in ROI},
                "target_2024_rb_upper": int(m.get("target_2024_rb_upper", 0)),
                "control_2024_rb_lower": int(m.get("control_2024_rb_lower", 0)),
                "control_2024_rb_distant": int(m.get("control_2024_rb_distant", 0)),
                "control_2026_rb_upper": int(m.get("control_2026_rb_upper", 0)),
                "duplicate_flag": int(bool(s["dup_of"])),
                "duplicate_of_sequence": s["dup_of"],
                # §3.4 기록 요구
                "selection_reason": "|".join(sorted(m["_reasons"])),
                "minisequence_len": len(s["members"]),
                "has_grade_transition": int("transition" in seq_reasons),
                "has_context_image": int("anchor" in seq_reasons),
                "is_context_anchor": int(m["_reasons"] == {"anchor"}),
                "patient_total_images": m["_patient_total"],
                # §3.5 라벨 출처
                "label_source": "reread",
                "patient_source_uniform": int(len(s["members"]) == m["_patient_total"]),
            })
    return reading, admin, manifest


def main():
    rng = random.Random(SEED)
    labels = read_csv(LABELS_CSV)
    if not labels:
        raise SystemExit(f"labels 비어 있음: {LABELS_CSV}")

    cols = list(labels[0].keys())
    for c in (UID_COL, PATIENT_COL, TARGET_ROI, *ROI):
        if c not in cols:
            raise SystemExit(f"필수 컬럼 없음: {c}")
    time_cols = [c for c in TIME_COLS if c in cols]
    has_year, has_image = YEAR_COL in cols, IMAGE_COL in cols

    rows = []
    for row in labels:
        r = dict(row)
        r["_uid"] = str(row[UID_COL])
        r["_patient"] = str(row[PATIENT_COL])
        r["_year"] = infer_year(row, has_year)
        r["_grade"] = int_grade(row[TARGET_ROI])
        r["_time_cols"] = time_cols
        rows.append(r)

    store, pat_counts = select_cases(rows, rng)
    if not store:
        raise SystemExit("선정된 영상 없음 — patient_id 접두어(24_/26_)와 RB 컬럼 확인")
    seqs, dups = build_sequences(store, rng)
    reading, admin, manifest = emit(order_sequences(seqs, dups, rng), has_image)

    write_csv(OUT_DIR / "p2a_reading_sheet.csv", reading)
    write_csv(OUT_DIR / "p2a_admin_key.csv", admin)
    write_csv(OUT_DIR / "p2a_image_manifest.csv", manifest)

    def count(flag):
        rr = [r for r in admin if int(r[flag]) and not r["duplicate_flag"]]
        return {"images": len(rr), "patients": len({r["patient_id"] for r in rr})}

    pool = Counter()
    for r in rows:
        if r["_year"] and r["_grade"] is not None:
            pool[(r["_year"], r["_grade"])] += 1

    n_dup = sum(r["duplicate_flag"] for r in admin)
    n_unique = len(store)
    summary = {
        "labels": str(LABELS_CSV),
        "target_full_enumeration": TARGET_FULL_ENUMERATION,
        "n_original_rows": len(labels),
        "n_selected_unique_images": n_unique,
        "n_reading_rows_including_duplicates": len(reading),
        "n_hidden_duplicate_images": n_dup,
        # 비율 기준 = 원본 판독 대상(중복 제외). quota 와 같은 분모다.
        "duplicate_frac_target": DUPLICATE_FRAC,
        "duplicate_frac_actual": round(n_dup / max(n_unique, 1), 4),
        "n_sequences": len(seqs) + len(dups),
        "patients_per_group": pat_counts,
        "time_columns_used": time_cols,
        "groups": {f: count(f) for f in (
            "target_2024_rb_upper", "control_2024_rb_lower",
            "control_2024_rb_distant", "control_2026_rb_upper")},
        "pool_2024_rb_upper": pool[("2024", 3)] + pool[("2024", 4)],
        "pool_2026_rb_upper": pool[("2026", 3)] + pool[("2026", 4)],
        "rb_grade_counts_by_year": {
            y: {str(g): pool[(y, g)] for g in range(5)} for y in ("2024", "2026")},
    }
    (OUT_DIR / "p2a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report(summary, admin)


def report(summary, admin):
    """Result/ 산출과 같은 내용을 표로 출력한다 (check_value_*.py 규약)."""
    W = 76
    print("=" * W)
    print(f"build_minisequence — Phase 2A §3.2 대상군 · §3.4 mini-sequence")
    print("=" * W)

    print("\n  [RB 기존 등급 분포]")
    print(f"    {'연도':<8}" + "".join(f"{'g'+str(g):>7}" for g in range(5)) + f"{'합':>8}")
    for y in ("2024", "2026"):
        c = summary["rb_grade_counts_by_year"][y]
        print(f"    {y:<8}" + "".join(f"{c[str(g)]:>7}" for g in range(5))
              + f"{sum(c.values()):>8}")

    print("\n  [§3.2 대상군]  선정 / 모집단")
    print(f"    {'대상군':<26}{'영상':>7}{'환자':>7}{'모집단':>9}")
    pools = {"target_2024_rb_upper": summary["pool_2024_rb_upper"],
             "control_2026_rb_upper": summary["pool_2026_rb_upper"]}
    names = {"target_2024_rb_upper": "표적군 2024 RB 3·4",
             "control_2024_rb_lower": "하위경계대조 2024 RB 2·3",
             "control_2024_rb_distant": "원거리대조 2024 RB 0·1",
             "control_2026_rb_upper": "연도대조 2026 RB 3·4"}
    for f, label in names.items():
        g = summary["groups"][f]
        pool = pools.get(f, "")
        print(f"    {label:<26}{g['images']:>7}{g['patients']:>7}{str(pool):>9}")

    dup = summary["n_hidden_duplicate_images"]
    tot = summary["n_reading_rows_including_duplicates"]
    uniq = summary["n_selected_unique_images"]
    # 비율 분모는 원본 판독 대상(중복 제외). quota 와 같은 기준이다.
    print(f"    {'hidden duplicate':<26}{dup:>7}{'':>7}"
          f"{summary['duplicate_frac_actual'] * 100:>8.1f}%")
    print(f"      목표 {summary['duplicate_frac_target'] * 100:.0f}% "
          f"(원본 {uniq}장 기준, 초과 금지) → 실제 {dup}장")

    pc = summary["patients_per_group"]
    print(f"\n  [대조군 환자 층화]  상한 {pc['control_cap']}명 (= 표적군 환자 수)")
    print(f"    표적군 {pc['target']}명 · 연도대조 {pc['year_ctl']}명 · "
          f"하위대조 {pc['lower_ctl']}명 · 원거리대조 {pc['distant_ctl']}명")

    print("\n  [§3.4 mini-sequence]")
    lens = Counter(int(a["minisequence_len"]) for a in admin)
    print(f"    시퀀스 길이 분포  {dict(sorted(lens.items()))}")
    trans = len({a["sequence_id"] for a in admin if int(a["has_grade_transition"])})
    ctx = len({a["sequence_id"] for a in admin if int(a["has_context_image"])})
    print(f"    등급 전이 포함 시퀀스  {trans} / {summary['n_sequences']}")
    print(f"    문맥 영상 포함 시퀀스  {ctx} / {summary['n_sequences']}")
    print(f"    표적군 전수 열거(TARGET_FULL_ENUMERATION)  {summary['target_full_enumeration']}")

    print(f"\n  고유 영상 {summary['n_selected_unique_images']}장 · "
          f"총 판독 {tot}장 · 시퀀스 {summary['n_sequences']}개 · "
          f"ROI 판정 {tot * 4:,}건")
    print("\n" + "=" * W)
    print(f"[save] {OUT_DIR}")
    for n in ("p2a_reading_sheet.csv", "p2a_admin_key.csv",
              "p2a_image_manifest.csv", "p2a_summary.json"):
        print(f"  - {n}")
    print("=" * W)


if __name__ == "__main__":
    main()
