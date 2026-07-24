# -*- coding: utf-8 -*-
r"""
A1 검증 실험 — 공용 코어 (draft.md §5.2 / A1_검증실험계획 §2·§3)

전략: 무거운 DORGA 학습 파이프라인(dataset·model·train loop·bootstrap)은
`dorga_train_2026to2024.py`(4-arm bias-direction 드라이버)에 이미 있으므로,
그 모듈을 그대로 import 해서 코어로 쓴다. 우리가 바꾸는 건 **split 생성(make_split)뿐**이다.

  - B.run_one(mode, seed, epochs, cache2026, root) 는 내부에서
        df = read_csv(MANIFEST); df, ty, ey = make_split(df, mode, seed)
    를 호출한다. 우리는 B.make_split 을 디스패처로 교체(monkeypatch)해서,
    등록된 mode 이름에 대해 우리의 커스텀 split(+라벨 주입)을 반환하게 한다.
  - 커스텀 splitter 는 df 의 'split' 컬럼(train/val/test)을 채우고,
    필요하면 ROI 라벨 컬럼을 그 자리에서 수정(E3/E4 주입)한 뒤 반환한다.
    run_one 하류의 gen_patterns/compute_priors/loss 는 전부 그 df 를 그대로 쓴다.

이 파일은 실행 진입점이 아니다. E1~E4 드라이버가 import 해서 쓴다.
"""
from __future__ import annotations
import os, sys, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

# ── Windows cp949 콘솔에서 base 스크립트의 유니코드 출력(✔ ━ 등) 크래시 방지 ──
#   base 로드(=wandb import) 전에 콘솔 캡처를 끄고 stdout 을 utf-8 로 재설정한다.
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

# ── 학습 코어: Private Experiment/core.py (舊 dorga_train_2026to2024.py 자립 대체) ──
_HERE = os.path.dirname(os.path.abspath(__file__))     # Experiment/E
_EXP_ROOT = os.path.dirname(_HERE)                     # Experiment
_REPO = os.path.dirname(_EXP_ROOT)                     # Private repo 루트
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

import core as B                                       # noqa: E402  학습 엔진(split/dataset/train/eval)

# early stopping: val MAE 가 10 에폭 개선 없으면 중단 (전 arm 공통)
EARLYSTOP = 10
B.EARLYSTOP_PATIENCE = EARLYSTOP

# 자주 쓰는 심볼 재노출 — ROI 순서는 core 기준 [RT, LT, RB, LB] 강제
ROI      = B.ROI                 # ["RT","LT","RB","LB"]
C        = B.C                   # 5
MANIFEST = B.CSV_PATH            # labels.csv (uid, patient_id, RT,LT,RB,LB, image_path)
DEVICE   = B.DEVICE
A1_OUT   = Path(_REPO) / "checkpoint"        # A1 산출 루트 → checkpoint/{A,L,E}/ (gitignore)
A1_OUT.mkdir(parents=True, exist_ok=True)

# ── make_split 몽키패치: mode 이름 → 등록된 splitter ─────────────
_ORIG_make_split = B.make_split
_SPLITTERS: dict = {}             # mode(str) -> fn(df, seed) -> (df_split, train_year, test_year)

def _prep(df):
    """core labels.csv(patient_id 24_/26_ 접두어) → 구 splitter 호환 year·patient 파생."""
    if "year" not in df.columns:
        df = df.copy()
        df["patient"] = df[B.PATIENT_COL]
        df["year"] = df[B.PATIENT_COL].astype(str).str[:2].map({"24": 2024, "26": 2026})
    return df

def _dispatch(df, mode, seed):
    df = _prep(df)
    if mode in _SPLITTERS:
        return _SPLITTERS[mode](df, seed)
    return _ORIG_make_split(df, mode, seed)

B.make_split = _dispatch

def register(mode: str, fn):
    """mode 이름에 커스텀 splitter 를 건다. run_one(mode,...) 가 이걸 타게 된다."""
    _SPLITTERS[mode] = fn


# ═══════════════════════════════════════════════════════════════
# 2026 마스크 캐시 (in-domain2026·matched 등 2026 영상이 split 에 들어갈 때만 필요)
# ═══════════════════════════════════════════════════════════════
# core 는 사전정렬 images_normalize 를 디스크에서 읽으므로 2026 in-code 캐시가 불필요.
def build_full_2026_cache():
    return {}                     # 하위호환 no-op (舊 seg+STN 캐시 제거)

EMPTY_CACHE: dict = {}            # 2024-only 실험용 (2026 영상 없음)


# ═══════════════════════════════════════════════════════════════
# split helper — 환자 단위 (누수 방지: 한 환자는 한 split 에만)
# ═══════════════════════════════════════════════════════════════
def _patients_of(df, year):
    return np.asarray(df.loc[df.year == year, "patient"].unique(), dtype=object)

def _mark(df, train_pat, val_pat, test_pat, year):
    """df 에서 해당 year 행만 남기고 split 을 채운다."""
    sub = df[df.year == year].copy()
    tr, va, te = set(train_pat), set(val_pat), set(test_pat)
    sub["split"] = np.where(sub.patient.isin(te), "test",
                    np.where(sub.patient.isin(va), "val",
                    np.where(sub.patient.isin(tr), "train", None)))
    return sub[sub.split.notna()].copy()

def kfold_patient_folds(df, year, n_folds, seed):
    """year 코호트 환자를 n_folds 로 분할 → [(fold_idx, test_patients_array), ...]"""
    pats = _patients_of(df, year)
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    return [(k, pats[k::n_folds]) for k in range(n_folds)]   # 인터리브 분할 (분포 균등)


# ═══════════════════════════════════════════════════════════════
# 라벨 주입 (E3 양성대조) — 순서형 오프셋, 천장/바닥 clip
#   b_half 오프셋을 '그 half 에 속한 모든 행'에 일관 적용한다(학습·평가 무관).
#   분수 β 는 stochastic rounding 으로 표현.  clip(0,C-1) 이라 극단 등급에서
#   복원 기울기가 감쇠하는데, 그 감쇠 자체가 E3 복원곡선으로 측정된다.
# ═══════════════════════════════════════════════════════════════
def inject_offset(df, mask, beta, seed):
    """mask(bool Series/array) 인 행의 ROI 라벨에 오프셋 beta 를 순서형으로 적용."""
    if beta == 0:
        return df
    rng = np.random.default_rng(seed)
    df = df.copy()
    idx = np.where(np.asarray(mask))[0]
    for col in ROI:
        g = df[col].to_numpy().astype(float)
        base = g[idx] + beta
        lo = np.floor(base)
        up = (rng.random(len(idx)) < (base - lo)).astype(float)
        g[idx] = np.clip(lo + up, 0, C - 1)
        df[col] = g.astype(int)
    return df


# ═══════════════════════════════════════════════════════════════
# 분포 정합 (E2) — 두 코호트를 공통 등급분포·공통 크기로 서브샘플
#   ROI-평균 등급을 환자 대리로 쓰고, 등급 히스토그램을 맞춰 환자를 표집한다.
# ═══════════════════════════════════════════════════════════════
def match_two_cohorts(df, seed, n_bins=5):
    """2024·2026 을 환자 단위로 공통 분포·공통 환자수로 맞춘 부분집합 반환."""
    rng = np.random.default_rng(seed)
    def pat_grade(year):
        sub = df[df.year == year]
        return sub.groupby("patient")[ROI].apply(lambda x: x.to_numpy().mean())
    g24, g26 = pat_grade(2024), pat_grade(2026)
    edges = np.linspace(0, C - 1, n_bins + 1)
    b24, b26 = np.digitize(g24.values, edges), np.digitize(g26.values, edges)
    keep24, keep26 = [], []
    for b in range(1, n_bins + 2):
        p24, p26 = g24.index[b24 == b], g26.index[b26 == b]
        k = min(len(p24), len(p26))                     # 각 bin 에서 min 만큼만
        if k == 0: continue
        keep24 += list(rng.choice(p24, k, replace=False))
        keep26 += list(rng.choice(p26, k, replace=False))
    return set(keep24), set(keep26)


# ═══════════════════════════════════════════════════════════════
# 실행 래퍼 — 등록된 mode 로 B.run_one 호출
# ═══════════════════════════════════════════════════════════════
def run_arm(mode, splitter, seed, epochs, cache=None, root=None, model="dorga", arm=None):
    """splitter 등록 후 core.train_arm 실행. cache 인자는 하위호환용(무시 — 사전정렬 디스크 로드).
    arm 주면 run_dir 이름을 그걸로(미지정 시 {mode}_s{seed}). results dict(+ npz) 반환."""
    register(mode, splitter)
    root = Path(root)
    B.train_arm(model, mode, seed, epochs, root, arm=arm)
    run_dir = root / (arm if arm else f"{mode}_s{seed}")
    res = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    res["npz"] = str(run_dir / "test_preds.npz")
    return res


# ═══════════════════════════════════════════════════════════════
# per-patient bias & δ/s 분해 + bootstrap CI
# ═══════════════════════════════════════════════════════════════
def per_patient_bias(npz_path, roi=None):
    """test_preds.npz → 환자별 bias 배열. roi=None 이면 4-ROI 평균, 아니면 해당 ROI."""
    d = np.load(npz_path, allow_pickle=True)
    P, Y, pats = d["preds"], d["labels"], np.asarray(d["patients"])
    if roi is None:
        e = (P - Y).mean(axis=1)
    else:
        j = ROI.index(roi); e = (P[:, j] - Y[:, j]).astype(float)
    uniq = np.unique(pats)
    return np.array([e[pats == u].mean() for u in uniq])

def mean_ci(arr, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return float(arr.mean()), tuple(np.percentile(boots, [2.5, 97.5]))

def diff_ci(a, b, n_boot=5000, seed=0):
    """독립 두 표본 평균차 a-b 의 bootstrap CI (fwd/rev 는 test 셋이 달라 비대응)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    boots = a[ia].mean(1) - b[ib].mean(1)
    return float(a.mean() - b.mean()), tuple(np.percentile(boots, [2.5, 97.5]))

def decompose(fwd_npz, rev_npz, roi=None, n_boot=5000, seed=0):
    """δ=(rev-fwd)/2 (라벨 성분), s=(rev+fwd)/2 (모델 성분) + 각 CI. fwd/rev 독립 bootstrap."""
    f = per_patient_bias(fwd_npz, roi)
    r = per_patient_bias(rev_npz, roi)
    rng = np.random.default_rng(seed)
    bf = f[rng.integers(0, len(f), (n_boot, len(f)))].mean(1)
    br = r[rng.integers(0, len(r), (n_boot, len(r)))].mean(1)
    delta, s = (br - bf) / 2, (br + bf) / 2
    return dict(fwd=float(f.mean()), rev=float(r.mean()),
                delta=float((r.mean() - f.mean()) / 2), delta_ci=tuple(np.percentile(delta, [2.5, 97.5])),
                s=float((r.mean() + f.mean()) / 2),     s_ci=tuple(np.percentile(s,     [2.5, 97.5])))

def sig(ci):     # CI 가 0 을 배제하는가
    return ci[0] * ci[1] > 0


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
