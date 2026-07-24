import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# inspect_data.py 로 확인된 실제 구조 (1305장)
#   파일명 컬럼 : 'uid'  (uid + '.png', 1305/1305 매칭)
#   점수 컬럼   : ['RT','LT','RB','LB'] 순서 그대로, 값 0~4 (5단계 -> classes=5)
#   환자 컬럼   : 'patient_id' (114명) -> 분할은 환자 단위로
FNAME_COL = "uid"
SCORE_COLS = ["RT", "LT", "RB", "LB"]
PATIENT_COL = "patient_id"
EXT = ".png"


# ----------------------------------------------------------------------
# 증강 헬퍼 (지정된 photometric 구성)
# ----------------------------------------------------------------------
class RandomGamma:
    """확률 p 로 감마 보정. 입력은 PIL(L) 또는 0~1 텐서 이전 단계의 PIL.
    ToTensor 앞에 두므로 PIL 이미지에 적용한다.
    """

    def __init__(self, gamma_range=(0.8, 1.2), p=0.2):
        self.lo, self.hi = gamma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() >= self.p:
            return img
        g = np.random.uniform(self.lo, self.hi)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0, 1) ** g
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")


class AddGaussianNoise:
    """확률 p 로 가우시안 노이즈. ToTensor 뒤(0~1 텐서)에 적용."""

    def __init__(self, mean=0.0, std=0.02, p=0.2):
        self.mean, self.std, self.p = mean, std, p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return t
        return t + torch.randn_like(t) * self.std + self.mean


def build_photometric(mode: str):
    """지정된 증강 구성. train 만 증강, 그 외는 ToTensor+Normalize 만.
    Normalize([0.56],[0.17]) 은 0~1 스케일 기준 통계이므로 ToTensor 뒤에 온다.
    """
    if mode == "train":
        return transforms.Compose([
            RandomGamma(gamma_range=(0.8, 1.2), p=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(p=0.2),
            transforms.Normalize([0.56], [0.17]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.56], [0.17]),
    ])


class InhaUHDataset(Dataset):
    def __init__(self, img_dir: str, csv_path: str,
                 fname_col: str = FNAME_COL,
                 score_cols: Optional[List[str]] = None,
                 img_size: int = 512, mode: str = "train",
                 rows: Optional[np.ndarray] = None,
                 mask_dir: Optional[str] = None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        self.transform = build_photometric(mode)
        self.mask_dir = mask_dir            # 지정되면 캐싱된 마스크를 함께 반환

        df = pd.read_csv(csv_path)
        if rows is not None:
            df = df.iloc[rows].reset_index(drop=True)
        self.df = df

        self.fname_col = fname_col
        self.score_cols = score_cols or SCORE_COLS
        assert len(self.score_cols) == 4, "score_cols 는 4개 [RT,LT,RB,LB]"

    def _resolve(self, uid: str) -> str:
        # uid 는 확장자가 없음 -> .png 를 붙임. 이미 붙어 있으면 그대로.
        return uid if uid.lower().endswith(EXT) else uid + EXT

    def __len__(self):
        return len(self.df)

    def _load_image(self, path) -> Image.Image:
        im = Image.open(path).convert("L")           # 흑백 1채널
        if im.size != (self.img_size, self.img_size):
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        return im                                     # PIL 그대로 (transform 이 텐서화)

    def _load_mask(self, uid: str) -> torch.Tensor:
        """캐싱된 512 마스크 (1,H,W) {0,1}. Normalize 대상 아님."""
        path = os.path.join(self.mask_dir, uid + EXT)
        m = Image.open(path).convert("L")
        if m.size != (self.img_size, self.img_size):
            m = m.resize((self.img_size, self.img_size), Image.NEAREST)
        arr = (np.asarray(m, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
        return torch.from_numpy(arr)[None]            # (1,H,W)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row[self.fname_col])
        fname = self._resolve(uid)
        img = self._load_image(os.path.join(self.img_dir, fname))
        img = self.transform(img)                     # (1,H,W), ToTensor+Normalize 포함

        y = np.array([row[c] for c in self.score_cols], dtype=np.int64)  # [RT,LT,RB,LB]
        target = torch.from_numpy(y).view(2, 2)       # [[RT,LT],[RB,LB]]

        if self.mask_dir is not None:
            mask = self._load_mask(uid)               # (1,H,W)
            return img, mask, target
        return img, target


def split_indices_by_patient(csv_path: str, val_frac: float = 0.15,
                             seed: int = 0):
    """환자 단위 분할. 한 환자의 여러 프레임이 train/val 에 걸치는 누출을 막는다.

    이미지 단위로 나누면 같은 환자(평균 11프레임)가 양쪽에 들어가 검증 성능이
    부풀려진다. 반드시 patient_id 로 나눈다.
    """
    df = pd.read_csv(csv_path)
    patients = df[PATIENT_COL].unique()
    patients = np.asarray(patients, dtype=object)   # StringArray shuffle 경고 회피
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_frac))
    val_pat = set(patients[:n_val])

    is_val = df[PATIENT_COL].isin(val_pat).values
    all_idx = np.arange(len(df))
    tr_idx, va_idx = all_idx[~is_val], all_idx[is_val]
    print(f"[split] 환자 {len(patients)}명 -> train {len(patients)-n_val} / "
          f"val {n_val},  이미지 train {len(tr_idx)} / val {len(va_idx)}")
    return tr_idx, va_idx