"""
BS-Net 스코어링 모델 (PyTorch) - 정렬된 이미지 입력용

전제:
  - 입력 이미지는 이미 STN 정렬이 끝난 상태 (images_normalize).
    따라서 모델 안에 STN warp 없음. feature 는 identity, ROI 는 고정 박스 크롭만.
  - hard attention 용 폐 마스크는 학습된 seg 모델이 정렬 이미지에 대해 뽑아준다.
    이미지가 정렬돼 있으니 마스크도 자동으로 정렬돼 있음.
  - 백본(ResNet-18)은 이 모델이 포함한다.

영역 배치 (2x2):
    [[RT, LT],
     [RB, LB]]
  열 0 = 우폐(영상 좌측), 열 1 = 좌폐(영상 우측)
  행 0 = 상, 행 1 = 하
"""

from typing import List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ======================================================================
# 영역 정의
# ======================================================================
def make_boxes(vertical_overlap: float = 0.25):
    """2x2 박스를 (y1, x1, y2, x2) 정규화 좌표로.

    vertical_overlap = 인접 두 행의 겹침 / 한 행의 높이.
      0.25 -> 행 높이 h: 2h - 1 = 0.25h => h = 4/7 ≈ 0.571
              행 = [0, 0.571], [0.429, 1.0]  (원본 Brixia 3행의 세로중첩 비율과 동일)
      0.0  -> 행 = [0, 0.5], [0.5, 1.0]  (중첩 없음)
    가로는 0.5 이분, 중첩 없음.
    """
    assert 0.0 <= vertical_overlap < 1.0
    h = 1.0 / (2.0 - vertical_overlap)
    rows = [(0.0, h), (1.0 - h, 1.0)]
    cols = [(0.0, 0.5), (0.5, 1.0)]
    boxes, names = [], []
    for r, (y1, y2) in zip(("T", "B"), rows):
        for c, (x1, x2) in zip(("R", "L"), cols):
            boxes.append((y1, x1, y2, x2))
            names.append(f"{c}{r}")            # RT, LT, RB, LB
    return boxes, names


def _roi_thetas(boxes) -> torch.Tensor:
    """정규화 박스 -> grid_sample affine (K,2,3). tf.crop_and_resize 와 동치."""
    th = torch.zeros(len(boxes), 2, 3)
    for k, (y1, x1, y2, x2) in enumerate(boxes):
        th[k, 0, 0] = x2 - x1
        th[k, 0, 2] = x1 + x2 - 1.0
        th[k, 1, 1] = y2 - y1
        th[k, 1, 2] = y1 + y2 - 1.0
    return th


def _warp(x, theta, out_hw):
    grid = F.affine_grid(theta, (x.size(0), x.size(1), *out_hw), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


class Swish(nn.Module):
    def forward(self, x):
        return F.silu(x)


# ======================================================================
# 백본 레지스트리
# ======================================================================
# 모든 백본이 지켜야 하는 계약:
#   - forward(img: (B, in_ch, H, W)) -> List[Tensor] 길이 4, coarse->fine 순서 아님,
#     [c1, c2, c3, c4] 로 해상도가 점점 작아지는 순서 (stride 4,8,16,32 권장)
#   - .out_channels : 길이 4 튜플, 각 스테이지 채널 수
# 이 둘만 지키면 FPN/ROIPool/헤드/손실은 그대로 재사용된다.
# (FPN in_ch 는 backbone.out_channels 로 자동 설정, ROIPool/attention 은 해상도 무관)

# --- bsnet 백본: ResNet-18 (원 BS-Net 백본) ------------------------------
class ResNet18Features(nn.Module):
    """512x512 입력 -> c1(64,128²) c2(128,64²) c3(256,32²) c4(512,16²)."""

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        # 흑백 입력이면 conv1 을 1채널로 교체하고, 사전학습 가중치는 채널 평균으로 이식
        if in_channels != 3:
            old = net.conv1
            new = nn.Conv2d(in_channels, old.out_channels,
                            kernel_size=old.kernel_size, stride=old.stride,
                            padding=old.padding, bias=old.bias is not None)
            if pretrained:
                with torch.no_grad():
                    w = old.weight.mean(dim=1, keepdim=True)     # (64,1,7,7)
                    new.weight.copy_(w.repeat(1, in_channels, 1, 1))
            net.conv1 = new

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = (64, 128, 256, 512)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


# --- pafe 백본 (미지정) --------------------------------------------------
class PAFEBackbone(nn.Module):
    """[작성 필요] PAFE 백본.

    아래 계약만 지키면 나머지 파이프라인은 그대로 동작한다:
      - __init__(in_channels, pretrained)
      - forward(img) -> [c1, c2, c3, c4]  (해상도 감소 순)
      - self.out_channels = (C1, C2, C3, C4)
    아키텍처 정의 파일을 알려주면 여기에 연결한다.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        raise NotImplementedError(
            "PAFE 백본 정의가 필요함. 4-스테이지 feature 를 내는 모듈을 여기 연결하라.")

    def forward(self, x):
        raise NotImplementedError


# --- dorga 백본 (미지정) -------------------------------------------------
class DORGABackbone(nn.Module):
    """[작성 필요] DORGA 백본. 계약은 PAFEBackbone 과 동일.

    DORGA 의 EncoderConv(seg.py 의 그래프 U-Net 인코더) 를 feature 추출기로
    쓸 생각이면, 그 인코더가 내는 중간 텐서 4개를 [c1,c2,c3,c4] 로 반환하도록
    래핑하면 된다. 다만 EncoderConv 는 stride/채널 구성이 ResNet 과 달라
    어떤 스테이지를 뽑을지 명시가 필요하다.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        raise NotImplementedError(
            "DORGA 백본 정의가 필요함. 어떤 4개 스테이지를 뽑을지 알려주면 연결한다.")

    def forward(self, x):
        raise NotImplementedError


BACKBONES = {
    "bsnet": ResNet18Features,
    "pafe": PAFEBackbone,
    "dorga": DORGABackbone,
}


def build_backbone(name: str, in_channels: int = 1, pretrained: bool = True):
    if name not in BACKBONES:
        raise ValueError(f"알 수 없는 백본 '{name}'. 선택지: {list(BACKBONES)}")
    return BACKBONES[name](in_channels=in_channels, pretrained=pretrained)


# ======================================================================
# ROI 풀링 (정렬 이미지 전제 -> STN warp 없음, 고정 박스 크롭만)
# ======================================================================
class ROIPool(nn.Module):
    def __init__(self, boxes, hard_attention: bool = True):
        super().__init__()
        self.hard_attention = hard_attention
        self.register_buffer("roi_theta", _roi_thetas(boxes))    # (R,2,3)

    @property
    def n_regions(self):
        return self.roi_theta.size(0)

    def forward(self, feats: List[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        R = self.n_regions
        out = []
        for c in feats:
            B, C, H, W = c.shape
            x = c
            if self.hard_attention:
                if mask is None:
                    raise ValueError("hard_attention=True 인데 mask 가 없음")
                m = F.interpolate(mask, size=(H, W), mode="bilinear",
                                  align_corners=False)
                x = x * m
            x = x.unsqueeze(1).expand(B, R, C, H, W).reshape(B * R, C, H, W)
            rt = self.roi_theta.unsqueeze(0).expand(B, R, 2, 3).reshape(B * R, 2, 3)
            out.append(_warp(x, rt, (H, W)))         # (B*R, C, H, W)
        return out


# ======================================================================
# FPN (원본 create_pyramid_features, 최종 P5 하나만 사용)
# ======================================================================
class BrixiaFPN(nn.Module):
    def __init__(self, in_ch=(64, 128, 256, 512), fs: int = 64):
        super().__init__()
        c1, c2, c3, c4 = in_ch
        self.c1_reduce = nn.Conv2d(c1, fs, 1)
        self.p1 = nn.Conv2d(fs, fs, 3, padding=1)
        self.c2_reduce = nn.Conv2d(c2, fs, 1)
        self.p2 = nn.Conv2d(fs, fs * 2, 3, padding=1)
        self.c3_reduce = nn.Conv2d(c3, fs * 2, 1)
        self.p3 = nn.Conv2d(fs * 2, fs * 4, 3, padding=1)
        self.c4_reduce = nn.Conv2d(c4, fs * 4, 1)
        self.p5 = nn.Conv2d(fs * 4, fs * 8, 3, padding=1)
        self.act = Swish()
        self.out_channels = fs * 8

    @staticmethod
    def _down(x, like):
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p = self.act(self.p1(self._down(self.c1_reduce(c1), c2)))
        p = self.act(self.p2(self._down(p + self.c2_reduce(c2), c3)))
        p = self.act(self.p3(self._down(p + self.c3_reduce(c3), c4)))
        p = self.act(self.p5(p + self.c4_reduce(c4)))
        return p


# ======================================================================
# 영역 헤드
# ======================================================================
class RegionHead(nn.Module):
    def __init__(self, in_ch, width=8, depth=3, classes=4):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(ch, width, 3, padding=1),
                       nn.BatchNorm2d(width), Swish()]
            ch = width
        self.body = nn.Sequential(*layers)
        self.score = nn.Conv2d(width, classes, 3, padding=1)

    def forward(self, x):
        return self.score(self.body(x)).mean(dim=(-2, -1))


# ======================================================================
# 전체 모델
# ======================================================================
class BSNet(nn.Module):
    """정렬된 CXR -> 2x2 영역 점수 logits (B,2,2,classes).

    mask 공급 방식 (hard_attention=True 일 때):
      A) forward(img, mask=...) 로 미리 뽑은 마스크를 직접 전달, 또는
      B) seg_model 을 생성자에 넘기면 forward 안에서 img 로 마스크를 계산.
         seg_model 은 (B,1,H,W) 입력 -> (B,1,H,W) 폐 확률(0~1) 을 반환해야 함.
    """

    def __init__(self, backbone: str = "bsnet", in_channels: int = 1,
                 classes: int = 5, hard_attention: bool = True,
                 vertical_overlap: float = 0.25, fs: int = 64,
                 head_width: int = 8, head_depth: int = 3,
                 pretrained_backbone: bool = True,
                 seg_model: Optional[Callable] = None):
        super().__init__()
        boxes, names = make_boxes(vertical_overlap)
        self.region_names = names
        self.classes = classes
        self.hard_attention = hard_attention
        self.backbone_name = backbone

        self.backbone = build_backbone(backbone, in_channels, pretrained_backbone)
        self.pool = ROIPool(boxes, hard_attention=hard_attention)
        self.fpn = BrixiaFPN(self.backbone.out_channels, fs=fs)  # 채널 자동 반영
        self.heads = nn.ModuleList([
            RegionHead(self.fpn.out_channels, head_width, head_depth, classes)
            for _ in range(len(boxes))
        ])

        # seg_model 은 학습 대상이 아님. nn.Module 을 그냥 대입하면 서브모듈로
        # 등록되어 state_dict/파라미터/train()/to() 에 딸려 들어가므로 등록을 우회한다.
        object.__setattr__(self, "_seg_model", seg_model)
        if seg_model is not None:
            for p in seg_model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _compute_mask(self, img):
        self._seg_model.eval()
        m = self._seg_model(img)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        return m.clamp(0, 1)

    def forward(self, img: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = img.size(0)

        if self.hard_attention and mask is None:
            if self._seg_model is None:
                raise ValueError(
                    "hard_attention=True 이면 mask 를 넘기거나 seg_model 을 지정해야 함")
            mask = self._compute_mask(img)

        feats = self.backbone(img)                   # 4 x (B,C,H,W)
        pooled = self.pool(feats, mask)              # 4 x (B*R,C,H,W)
        p5 = self.fpn(pooled)                        # (B*R, fs*8, 16, 16)
        R = self.pool.n_regions
        p5 = p5.view(B, R, *p5.shape[1:])

        logits = torch.stack([self.heads[i](p5[:, i]) for i in range(R)], dim=1)
        return logits.view(B, 2, 2, self.classes)    # [[RT,LT],[RB,LB]]

    # 단계별 학습용 freeze 헬퍼
    def freeze_backbone(self, flag=True):
        for p in self.backbone.parameters():
            p.requires_grad_(not flag)

    def set_backbone_bn_eval(self):
        """백본을 freeze 했을 때 BN 통계가 갱신되지 않도록."""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    B = 2
    img = torch.randn(B, 1, 512, 512)
    mask = torch.rand(B, 1, 512, 512)

    model = BSNet(backbone="bsnet", in_channels=1, classes=5,
                  hard_attention=True, pretrained_backbone=False)
    out = model(img, mask=mask)
    print("regions:", model.region_names)
    print("logits :", tuple(out.shape))
    n = sum(p.numel() for p in model.parameters())
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params : {n/1e6:.2f}M (trainable {nt/1e6:.2f}M)")