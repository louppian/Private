"""
CXR GUI V3 스코어링 모델 (PyTorch) - 통짜(단일 파일) 재구성

원본: D:\\CXR_GUI_V3\\modules\\{model.py, backbones.py, vit.py} 를 한 파일로 합침.
가중치: assets/weights/inha_all_weights_57.pth  (Hybrid · resnet34 · 5-class · 4-region)

전제 (BS-Net 파일과 동일한 정렬-입력 가정):
  - 입력 이미지는 이미 STN 정렬이 끝난 상태 (images_normalize). 모델 안에 STN warp 없음.
  - 원본 GUI 는 512x512 로 리사이즈 후 **3채널**(흑백을 3회 복제)로 넣는다.
    ImageNet 사전학습 백본과 저장 가중치가 3채널 기준이므로 이 모델도 3채널 입력을 받는다.
  - ROI 는 feature map 을 고정 박스로 자른 뒤 각 영역을 GAP → 분류하는 방식(하드 어텐션·seg 마스크 없음).

BS-Net 파일과의 차이 (구조가 다름, 의도적으로 원본 V3 를 그대로 보존):
  - FPN·seg 마스크 하드어텐션 없음.  백본 마지막 feature 하나만 쓴다.
  - 영역 헤드가 영역마다 따로가 아니라 **단일 fc 를 공유**한다(원본 그대로).
  - einops 제거: PatchEmbedding/MultiHeadAttention 의 rearrange 를 순수 torch 로 대체하되
    **서브모듈 이름·파라미터 키를 원본과 동일하게 유지**하여 inha_all_weights_57.pth 가 그대로 로드된다.

영역 순서 (num_regions==4, box=[y1,x1,y2,x2]):
  출력 = [RT, LT, RB, LB] (행 우선: 상행 R,L → 하행 R,L). 영상 좌측=환자 우폐(R).
  ※ 원본 V3 native 순서는 [RT,RB,LT,LB](A,B,C,D)였으나, GUI/BS-Net 규약에 맞춰 [RT,LT,RB,LB]로 재정렬.
    fc 가 전 영역 공유라 박스 순서 변경은 가중치 호환에 영향 없음.

서드파티 의존: torch, torchvision 만 (einops 불필요).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ======================================================================
# 백본 레지스트리 (원 backbones.py)
#   계약: 생성자() -> nn.Module,  model_dict[name] = [생성자, feature 채널수]
#   CNN/Hybrid 가 children()[:-2] 로 GAP/FC 를 떼어 [B, num_features, 16, 16] 을 만든다.
# ======================================================================
class ModifiedDensenet121(nn.Module):
    """densenet121 에서 avgpool·classifier 를 뗀 feature 추출기 (원본 그대로)."""

    def __init__(self):
        super(ModifiedDensenet121, self).__init__()
        self.densenet = torchvision.models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        self.features = nn.Sequential(*list(self.densenet.children())[:-1])

    def forward(self, x):
        return self.features(x)


def resnet18():
    return torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')


def resnet34():
    return torchvision.models.resnet34(weights='ResNet34_Weights.DEFAULT')


def resnet50():
    return torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')


def densenet121():
    return ModifiedDensenet121()


def mobilenet_v3_small():
    return torchvision.models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'densenet121': [densenet121, 1024],
    'mobilenet_v3_small': [mobilenet_v3_small, 576],
}


# ======================================================================
# ViT (원 vit.py) — einops 제거, 파라미터 키는 원본과 동일
# ======================================================================
class _ToTokens(nn.Module):
    """'b e h w -> b (h w) e'  (einops Rearrange 대체, 파라미터 없음)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)          # [B, HW, E]


class PatchEmbedding(nn.Module):
    """conv 패치화 + 학습 위치임베딩. projection[0]=Conv2d(키 보존), projection[1]=토큰화."""

    def __init__(self, in_channels: int = 3, patch_size: int = 16,
                 emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            _ToTokens(),
        )
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x + self.positions
        return x


class MultiHeadAttention(nn.Module):
    """원본 vit.py 의 어텐션을 수치 그대로 이식(스케일링 위치 등 원본 quirk 유지)."""

    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape
        H, d = self.num_heads, self.emb_size // self.num_heads
        # rearrange("b n (h d qkv) -> (qkv) b h n d")
        qkv = self.qkv(x).reshape(B, N, H, d, 3).permute(4, 0, 2, 1, 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]           # 각 [B,H,N,d]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(torch.float32).min)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling               # (원본과 동일: softmax 후 나눔)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)   # [B,H,N,d]
        # rearrange("b h n d -> b n (h d)")
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.emb_size)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0.0,
                 forward_expansion: int = 4, forward_drop_p: float = 0.0, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


# ======================================================================
# ROI 풀링 (원 pool_rois) — feature map 을 고정 박스로 자르고 각 영역을 crop_size 로 리사이즈
# ======================================================================
def region_boxes(num_regions: int):
    """box=[y1,x1,y2,x2] 정규화 좌표. 6=Brixia 세로중첩, 4=2x2 무중첩."""
    if num_regions == 6:
        return [
            [0.0, 0.0, 0.4, 0.5], [0.3, 0.0, 0.7, 0.5], [0.6, 0.0, 1.0, 0.5],   # A B C (좌폐열)
            [0.0, 0.5, 0.4, 1.0], [0.3, 0.5, 0.7, 1.0], [0.6, 0.5, 1.0, 1.0],   # D E F (우폐열)
        ]
    if num_regions == 4:
        # 출력 순서 = [RT, LT, RB, LB] (행 우선: 상행 R,L → 하행 R,L). 영상 좌측=환자 우폐(R).
        return [
            [0.0, 0.0, 0.5, 0.5],   # RT 좌상
            [0.0, 0.5, 0.5, 1.0],   # LT 우상
            [0.5, 0.0, 1.0, 0.5],   # RB 좌하
            [0.5, 0.5, 1.0, 1.0],   # LB 우하
        ]
    raise ValueError(f"num_regions must be 4 or 6, got {num_regions}")


def pool_rois(x: torch.Tensor, boxes, crop_size=None) -> torch.Tensor:
    """[B,C,H,W] → [B, R, C, *crop_size]. 각 박스를 잘라 crop_size 로 bilinear 리사이즈."""
    if crop_size is None:
        crop_size = x.shape[2:4]
    H, W = x.shape[2], x.shape[3]
    out = []
    for y1, x1, y2, x2 in boxes:
        crop = x[:, :, int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)]
        out.append(F.interpolate(crop, size=crop_size, mode='bilinear', align_corners=False))
    return torch.stack(out, dim=1)


# ======================================================================
# 모델: CNN (백본만) / Hybrid (백본 + ViT 1블록)  — 원 model.py
# ======================================================================
def _build_backbone_seq(backbone: str):
    """원본과 동일하게 백본을 nn.Sequential 로 감싸 GAP/FC 를 제거(키 보존 핵심)."""
    fn, num_features = model_dict[backbone]
    if backbone == 'densenet121':
        seq = nn.Sequential(fn())                                   # 이미 stripped
    else:
        seq = nn.Sequential(*list(fn().children())[:-2])           # avgpool·fc 제거
    return seq, num_features


class CNN(nn.Module):
    """정렬 CXR(3ch) → 영역별 클래스 확률 [B, R, num_classes]. (백본 + 공유 fc)"""

    def __init__(self, backbone: str, num_classes: int = 4, num_regions: int = 6):
        super(CNN, self).__init__()
        self.backbone, self.num_features = _build_backbone_seq(backbone)
        self.num_classes = num_classes
        self.num_regions = num_regions
        self.img_size = 16
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self._boxes = region_boxes(num_regions)

    def forward(self, x, embedding: bool = False):
        feat = self.backbone(x)                                     # [B, F, 16, 16]
        rois = pool_rois(feat, self._boxes)                         # [B, R, F, 16, 16]
        out = []
        for i in range(self.num_regions):
            pred = self.score_GAP(rois[:, i]).view(-1, self.num_features)   # [B, F]
            if not embedding:
                pred = F.softmax(self.fc(pred), dim=1)
            out.append(pred)
        return torch.stack(out, dim=1)                              # [B, R, num_classes]


class Hybrid(nn.Module):
    """정렬 CXR(3ch) → ViT hybrid → 영역별 클래스 확률 [B, R, num_classes]. (V3 GUI 본 모델)"""

    def __init__(self, backbone: str, num_classes: int = 4, num_regions: int = 6):
        super(Hybrid, self).__init__()
        self.backbone, num_features = _build_backbone_seq(backbone)
        self.num_classes = num_classes
        self.num_regions = num_regions
        self.img_size = 16
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1,
                                        emb_size=768, img_size=self.img_size)
        self.transformer = TransformerEncoderBlock()
        self.fc = nn.Linear(768, self.num_classes)
        self._boxes = region_boxes(num_regions)

    def forward(self, x, embedding: bool = False):
        feat = self.backbone(x)                                     # [B, F, 16, 16]
        tok = self.patch_emb(feat)                                  # [B, 256, 768]
        tok = self.transformer(tok)                                 # [B, 256, 768]
        feat = tok.reshape(-1, self.img_size, self.img_size, 768).permute(0, 3, 1, 2)
        rois = pool_rois(feat, self._boxes)                         # [B, R, 768, 16, 16]
        out = []
        for i in range(self.num_regions):
            pred = self.score_GAP(rois[:, i]).view(-1, 768)
            if not embedding:
                pred = F.softmax(self.fc(pred), dim=1)
            out.append(pred)
        return torch.stack(out, dim=1)                              # [B, R, num_classes]


# ======================================================================
# 빌드 헬퍼 (V3 GUI 기본 설정)
# ======================================================================
def build_v3_model(kind: str = "hybrid", backbone: str = "resnet34",
                   num_classes: int = 5, num_regions: int = 4,
                   weights_path: str = None, map_location: str = "cpu"):
    """V3 GUI 기본값(hybrid·resnet34·5cls·4reg)으로 모델 생성, 있으면 가중치 로드."""
    model = {"hybrid": Hybrid, "cnn": CNN}[kind](backbone, num_classes, num_regions)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state)                                # 키 동일 → strict 로드
    return model


if __name__ == "__main__":
    B = 2
    img = torch.randn(B, 3, 512, 512)                               # 3채널 입력(원본 GUI 규약)

    model = build_v3_model(kind="hybrid", backbone="resnet34",
                           num_classes=5, num_regions=4)
    model.eval()
    with torch.no_grad():
        out = model(img)                                           # softmax 확률
        pred = out.argmax(-1)                                      # 영역별 등급
    print("output :", tuple(out.shape), "(B, R, classes)")         # (2, 4, 5)
    print("argmax :", tuple(pred.shape), "-> order [RT, LT, RB, LB]")
    n = sum(p.numel() for p in model.parameters())
    print(f"params : {n/1e6:.2f}M")
