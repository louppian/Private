import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "unified_labels.csv")
DEFAULT_OUT = r"/shared/home/mai/JeongGeon/Private/CXR/Merged"


# ---------------------------------------------------------------- 로드 --------
def load_rows(csv_path):
    if not os.path.exists(csv_path):
        sys.exit(f"[중단] 입력 CSV 없음: {csv_path}")
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("[중단] CSV가 비어 있음")
    need = {"uid", "patient_id", "seq", "n_frames",
            "RT", "LT", "RB", "LB", "src_frame", "image_path"}
    missing = need - set(rows[0].keys())
    if missing:
        sys.exit(f"[중단] CSV 컬럼 누락: {sorted(missing)}")
    return rows


def resolve_src(row, src26):
    """image_path를 실제 복사 소스로 해석. src26='raw'면 2026 경로를 RAW_IMAGE(jpg)로 전환."""
    p = row["image_path"]
    if src26 == "raw" and row["patient_id"].startswith("26_"):
        # ...\NPZ_IMAGE\NN_xxxx.png  ->  ...\RAW_IMAGE\<src_frame 대문자>.jpg
        d = os.path.dirname(os.path.dirname(p))          # 케이스 폴더
        raw = os.path.join(d, "RAW_IMAGE", row["src_frame"].upper() + ".jpg")
        return raw
    return p


# ------------------------------------------------------------- 계획 수립 ------
def build_plan(rows, img_dir, src26):
    plan, errors, seen = [], [], {}
    for r in rows:
        src = resolve_src(r, src26)
        # 출력 확장자: 소스 확장자를 따른다(raw면 .jpg, 아니면 .png)
        ext = os.path.splitext(src)[1].lower() or ".png"
        dst_name = f'{r["uid"]}{ext}'
        if not os.path.exists(src):
            errors.append(("MISSING_SRC", r["uid"], src))
            continue
        if dst_name in seen:
            errors.append(("DUP_DST", r["uid"], f'{dst_name} <- {seen[dst_name]}'))
            continue
        seen[dst_name] = r["uid"]
        plan.append((src, os.path.join(img_dir, dst_name), r))
    return plan, errors


def human(nbytes):
    for u in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or u == "GB":
            return f"{nbytes:.1f}{u}"
        nbytes /= 1024


# ---------------------------------------------------------------- 실행 --------
def do_copy(plan, execute):
    copied = skipped = 0
    total_bytes = 0
    for src, dst, _ in plan:
        ssize = os.path.getsize(src)
        total_bytes += ssize
        if os.path.exists(dst) and os.path.getsize(dst) == ssize:
            skipped += 1
            continue
        if execute:
            shutil.copy2(src, dst)
        copied += 1
    return copied, skipped, total_bytes


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def write_labels(plan, out_dir, execute):
    out_csv = os.path.join(out_dir, "labels.csv")
    rows = []
    for _, dst, r in plan:
        r = dict(r)
        r["image_path"] = "images/" + os.path.basename(dst)   # 상대경로로 재작성
        rows.append(r)
    cols = ["uid", "patient_id", "seq", "n_frames",
            "RT", "LT", "RB", "LB", "src_frame", "image_path"]
    if execute:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return out_csv, rows


def write_manifest(out_dir, args, plan, total_bytes, execute):
    n24 = sum(1 for _, _, r in plan if r["patient_id"].startswith("24_"))
    n26 = sum(1 for _, _, r in plan if r["patient_id"].startswith("26_"))
    info = {
        "source_csv": os.path.abspath(args.csv),
        "src26": args.src26,
        "out_dir": os.path.abspath(out_dir),
        "n_images": len(plan),
        "n_2024": n24,
        "n_2026": n26,
        "n_patients_2024": len({r["patient_id"] for _, _, r in plan if r["patient_id"].startswith("24_")}),
        "n_patients_2026": len({r["patient_id"] for _, _, r in plan if r["patient_id"].startswith("26_")}),
        "total_source_bytes": total_bytes,
        "note": "reader-final labels only (no model inference / posthoc). filename == uid.",
    }
    if execute:
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    return info


# ---------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser(description="2024+2026 CXR 통합 데이터셋 폴더 구축")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="입력 unified_labels.csv")
    ap.add_argument("--out", default=DEFAULT_OUT, help="출력 폴더")
    ap.add_argument("--src26", choices=["npz", "raw"], default="npz",
                    help="2026 이미지 소스: npz(정규화 png, 기본) | raw(원본 jpg)")
    ap.add_argument("--execute", action="store_true",
                    help="실제 복사 수행(미지정 시 DRY-RUN)")
    ap.add_argument("--verify-hash", action="store_true",
                    help="복사 후 샘플(최대 20개) SHA256 대조")
    args = ap.parse_args()

    execute = args.execute
    img_dir = os.path.join(args.out, "images")

    print(f"=== build_unified_dataset ({'EXECUTE' if execute else 'DRY-RUN'}) ===")
    print(f"  CSV   : {args.csv}")
    print(f"  OUT   : {args.out}")
    print(f"  src26 : {args.src26}\n")

    rows = load_rows(args.csv)
    plan, errors = build_plan(rows, img_dir, args.src26)

    # 유일성 재확인
    uids = [r["uid"] for r in rows]
    if len(set(uids)) != len(uids):
        dup = [u for u, c in Counter(uids).items() if c > 1]
        sys.exit(f"[중단] uid 중복: {dup[:10]}")

    print(f"계획 {len(plan)}건 · 오류 {len(errors)}건")
    if errors:
        for e in errors[:20]:
            print("   ", e)
        sys.exit("[중단] 소스 누락 또는 목적지 충돌. 복사하지 않음(fail-fast).")

    if execute:
        os.makedirs(img_dir, exist_ok=True)

    copied, skipped, total_bytes = do_copy(plan, execute)
    print(f"복사 대상 {copied} · skip(기존동일) {skipped} · 소스총량 {human(total_bytes)}")

    out_csv, _ = write_labels(plan, args.out, execute)
    info = write_manifest(args.out, args, plan, total_bytes, execute)
    print(f"  2024 {info['n_2024']}장/{info['n_patients_2024']}명 · "
          f"2026 {info['n_2026']}장/{info['n_patients_2026']}명")

    # 검증
    if execute:
        n_files = len([f for f in os.listdir(img_dir)
                       if os.path.isfile(os.path.join(img_dir, f))])
        ok = (n_files == len(plan))
        print(f"[검증] images={n_files} · labels={len(plan)} · 일치={ok}")
        print(f"[검증] labels.csv -> {out_csv}")
        if args.verify_hash:
            step = max(1, len(plan) // 20)
            bad = [r["uid"] for src, dst, r in plan[::step]
                   if sha256(src) != sha256(dst)]
            print(f"[검증] 해시 대조 {len(plan[::step])}개 · 불일치 {len(bad)} {bad[:5]}")
        if not ok:
            sys.exit("[경고] 파일수와 라벨행수 불일치")
    else:
        print("\n(DRY-RUN) 실제 복사·파일생성 없음. 확인 후 --execute 로 실행하세요.")


if __name__ == "__main__":
    main()