# -*- coding: utf-8 -*-
"""E0 best_val 체크포인트를 재추론해 방향반전 δ/γ 를 best_val 기준으로 재산출.
베이스 파이프라인(dorga_train_2026to2024) 그대로 사용 → 레시피 일치.
환자 부트스트랩 95% + Bonferroni(m=5, 99%) CI."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "E"))
import numpy as np, pandas as pd, torch
import e_common as A
B = A.B
E0 = r"D:\npjDM2026\runs\E0"
SEEDS = [1, 2, 42]
ROI = B.ROI                       # native ["RT","RB","LT","LB"]
DISP = ["RT", "LT", "RB", "LB"]

print("[1] 2026 seg+STN 캐시 빌드…")
cache = A.build_full_2026_cache()

@torch.no_grad()
def eval_arm(mode, seed):
    df = pd.read_csv(B.MANIFEST)
    df, TY, EY = B.make_split(df, mode, seed)
    df = B.gen_patterns(df, seed)
    pi_g, pi_p = B.compute_priors(df)
    vit = B.load_mae_ckpt_to_512(str(B.MRM_W), num_classes=B.C, in_chans=1, drop_path_rate=0.1, verbose=False)
    model = B.BrixiaViT512Dynamic(vit, num_regions=B.R, num_classes=B.C, num_patterns=B.K,
                                  proj_dim=B.PROJ_DIM, pi_global=pi_g, pi_patterns=pi_p,
                                  gnn_num_heads=4, gnn_dropout=0.1).to(B.DEVICE)
    ck = torch.load(f"{E0}/{mode}_s{seed}/best_val_model.pth", map_location=B.DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    from torch.utils.data import DataLoader
    te = df[df.split == "test"]
    loader = DataLoader(B.InhaUHMaskDataset(te, cache, "eval"), batch_size=B.BATCH_SIZE, num_workers=0)
    acc, mae, bias, per, P, Y = B.evaluate(model, loader)
    del model, vit
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return P, Y, te.patient.to_numpy().astype(str)

res = {}
for mode in ["2024to2026", "2026to2024"]:
    Ps, Ys, PTs = [], [], []
    for s in SEEDS:
        print(f"[eval] {mode} s{s}")
        P, Y, PT = eval_arm(mode, s)
        Ps.append(P); Ys.append(Y); PTs.append(PT)
    res[mode] = (np.vstack(Ps), np.vstack(Ys), np.concatenate(PTs))

fP, fY, fPT = res["2024to2026"]   # fwd: test 2026
rP, rY, rPT = res["2026to2024"]   # rev: test 2024
rng = np.random.default_rng(0)

def pat_means(P, Y, PT, col):
    b = (P[:, col] - Y[:, col]).astype(float) if col is not None else (P - Y).astype(float).mean(1)
    u = np.unique(PT); return u, np.array([b[PT == p].mean() for p in u])

def boot(u, pb, n=5000):
    return pb[rng.integers(0, len(u), size=(n, len(u)))].mean(1)

print("\n=== best_val 재산출 (E0 재추론, 3-seed) ===")
print(f"{'ROI':>7} {'fwd':>8} {'rev':>8} {'δ':>8} {'γ':>8} {'δ 95%CI':>18} {'δ Bonf99%':>18}")
for name in ["overall"] + DISP:
    col = None if name == "overall" else ROI.index(name)
    uf, pf = pat_means(fP, fY, fPT, col); ur, pr = pat_means(rP, rY, rPT, col)
    fwd, rev = pf.mean(), pr.mean(); delta, gamma = (rev - fwd) / 2, (rev + fwd) / 2
    bd = (boot(ur, pr) - boot(uf, pf)) / 2
    lo, hi = np.percentile(bd, [2.5, 97.5]); loB, hiB = np.percentile(bd, [0.5, 99.5])
    print(f"{name:>7} {fwd:>+8.3f} {rev:>+8.3f} {delta:>+8.3f} {gamma:>+8.3f} "
          f"[{lo:+.3f},{hi:+.3f}] [{loB:+.3f},{hiB:+.3f}]")
print("\n[참고] §4.4(최종모델): fwd −0.231 rev +0.004 γ −0.113 | RB δ+0.198 γ−0.151")
