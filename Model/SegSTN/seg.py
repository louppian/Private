#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seg.py — GUNet 폐 분할기 (LungSegmenter) 완전 이식본

원본 3개 파일을 하나로 인라인:
  - C:\\Code\\DORGA\\GUNet\\utils.py          (그래프 행렬 유틸)
  - C:\\Code\\DORGA\\GUNet\\GUNet_Utils.py    (ChebConv / Pool / residualBlock)
  - C:\\Code\\DORGA\\GUNet\\GUNet_model.py    (GUNet)
  - C:\\Code\\DORGA\\dorga\\preprocessing\\segmentation.py (LungSegmenter)
가중치: finetuned_9601.pt (= CXR_GUI_V4/assets/weights/finetuned_9601.pt, GUNet seg)

서드파티 의존(pip 설치 필요): torch, torchvision, opencv-python(cv2), scipy, numpy
  * torch_geometric(PyG)는 더 이상 필요 없음 — ChebConv/Pool을 순수 torch로 재구현했다
    (PyG 2.8의 propagate 시그니처가 이 코드와 비호환이라 버전 의존을 제거).

state_dict 호환을 위해 모든 클래스/속성 이름은 원본과 동일하게 유지한다.
"""

import cv2
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align


# ═══════════════════════════════════════════════════════════════════
# 1) 그래프 행렬 유틸 (원 GUNet/utils.py)
# ═══════════════════════════════════════════════════════════════════
def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(indices=i, values=v, size=scp_matrix.shape, dtype=torch.float32)


def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i - 1] = 1
        sub[i, (i + 1) % N] = 1
    return sub


def mOrganD(N):
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N2, N])
    for i in range(0, N2):
        if (2 * i + 1) == N:
            sub[i, 2 * i] = 1
        else:
            sub[i, 2 * i] = 1 / 2
            sub[i, 2 * i + 1] = 1 / 2
    return sub


def mOrganU(N):
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N, N2])
    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i // 2] = 1
        else:
            sub[i, i // 2] = 1 / 2
            sub[i, (i // 2 + 1) % N2] = 1 / 2
    return sub


def genMatrixesLungsHeart():
    RLUNG, LLUNG, HEART = 44, 50, 26
    Asub1, Asub2, Asub3 = mOrgan(RLUNG), mOrgan(LLUNG), mOrgan(HEART)
    Dsub1, Dsub2, Dsub3 = mOrganD(RLUNG), mOrganD(LLUNG), mOrganD(HEART)
    Usub1, Usub2, Usub3 = mOrganU(RLUNG), mOrganU(LLUNG), mOrganU(HEART)

    p1 = RLUNG; p2 = p1 + LLUNG; p3 = p2 + HEART
    p1_ = int(np.ceil(RLUNG / 2)); p2_ = p1_ + int(np.ceil(LLUNG / 2)); p3_ = p2_ + int(np.ceil(HEART / 2))

    A = np.zeros([p3, p3])
    A[:p1, :p1] = Asub1; A[p1:p2, p1:p2] = Asub2; A[p2:p3, p2:p3] = Asub3

    AD = np.zeros([p3_, p3_])
    AD[:p1_, :p1_] = mOrgan(int(np.ceil(RLUNG / 2)))
    AD[p1_:p2_, p1_:p2_] = mOrgan(int(np.ceil(LLUNG / 2)))
    AD[p2_:p3_, p2_:p3_] = mOrgan(int(np.ceil(HEART / 2)))

    D = np.zeros([p3_, p3])
    D[:p1_, :p1] = Dsub1; D[p1_:p2_, p1:p2] = Dsub2; D[p2_:p3_, p2:p3] = Dsub3

    U = np.zeros([p3, p3_])
    U[:p1, :p1_] = Usub1; U[p1:p2, p1_:p2_] = Usub2; U[p2:p3, p2_:p3_] = Usub3
    return A, AD, D, U


# ═══════════════════════════════════════════════════════════════════
# 2) 그래프 컨볼루션 블록 (원 GUNet/GUNet_Utils.py) — 순수 torch 재구현(PyG 무의존)
# ═══════════════════════════════════════════════════════════════════
class ChebConv(nn.Module):
    """PyG ChebConv(normalization='sym', lambda_max=2)의 무의존 재구현.

    Chebyshev 스펙트럴 그래프 컨볼루션:
        out = Σ_k lins[k]( T_k(L_hat) · x ) + bias,
        T_0=x, T_1=L_hat·x, T_k = 2·L_hat·T_{k-1} − T_{k-2}.
    sym 정규화 + lambda_max=2 에서 (자기루프 +1,−1 상쇄로) L_hat = −D^{-1/2} A D^{-1/2}, 대각 0.
    edge_weight=None(원본과 동일)이라 모든 간선 가중치 1. 노드 수 N이 작아(≤~120) dense 계산.
    state_dict 키(lins.k.weight, bias)를 PyG 원본과 동일하게 유지 → 가중치 그대로 로드.
    """

    def __init__(self, in_channels, out_channels, K, normalization='sym', bias=True):
        super().__init__()
        assert K > 0
        self.lins = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)]
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def _matmul(L_hat, x):
        # L_hat[N,N] · x → x가 [B,N,C]면 배치별, [N,C]면 단일
        if x.dim() == 2:
            return L_hat @ x
        return torch.einsum('nm,bmc->bnc', L_hat, x)

    def forward(self, x, edge_index, edge_weight=None):
        N = x.size(-2)
        A = torch.zeros(N, N, device=x.device, dtype=x.dtype)
        A[edge_index[0], edge_index[1]] = 1.0                 # 대칭 인접행렬(간선 1)
        deg = A.sum(dim=1)
        dis = deg.pow(-0.5)
        dis[torch.isinf(dis)] = 0.0
        L_hat = -(dis.view(-1, 1) * A * dis.view(1, -1))      # = −D^{-1/2} A D^{-1/2}

        out = self.lins[0](x)                                 # T_0 = x
        if len(self.lins) > 1:
            Tx_0 = x
            Tx_1 = self._matmul(L_hat, x)                     # T_1 = L_hat·x
            out = out + self.lins[1](Tx_1)
            for lin in self.lins[2:]:
                Tx_2 = 2.0 * self._matmul(L_hat, Tx_1) - Tx_0
                out = out + lin(Tx_2)
                Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out = out + self.bias
        return out


class Pool(nn.Module):
    """그래프 업샘플 out = U @ x.

    원본은 PyG MessagePassing.propagate(flow='source_to_target')로 U@x를 계산했으나,
    PyG 버전에 따라 propagate() 시그니처가 달라 'unexpected keyword x' 에러가 난다.
    파라미터가 없는 순수 선형연산이므로 sparse→dense 행렬곱으로 직접 구현(결과 동일, 버전 무관).
      pool_mat = U (sparse [N, N2]),  x [B, N2, C]  →  out [B, N, C].
    """
    def forward(self, x, pool_mat, dtype=None):
        U = pool_mat.to_dense().to(dtype=x.dtype, device=x.device)   # [N, N2]
        if x.dim() == 2:                                             # [N2, C]
            return U @ x
        return torch.einsum('nm,bmc->bnc', U, x)                     # [B, N2, C] -> [B, N, C]


class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
            self.skip = None
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.skip is not None:
            identity = self.skip(x)
        out += identity
        return F.relu(out)


# ═══════════════════════════════════════════════════════════════════
# 3) GUNet 모델 (원 GUNet/GUNet_model.py)
# ═══════════════════════════════════════════════════════════════════
class EncoderConv(nn.Module):
    def __init__(self, latents=64, hw=32):
        super(EncoderConv, self).__init__()
        self.latents = latents
        self.c = 4
        self.size = self.c * np.array([2, 4, 8, 16, 32], dtype=np.intc)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_down1 = residualBlock(1, self.size[0])
        self.dconv_down2 = residualBlock(self.size[0], self.size[1])
        self.dconv_down3 = residualBlock(self.size[1], self.size[2])
        self.dconv_down4 = residualBlock(self.size[2], self.size[3])
        self.dconv_down5 = residualBlock(self.size[3], self.size[4])
        self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        self.fc_mu = nn.Linear(in_features=self.size[4] * hw * hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.size[4] * hw * hw, out_features=self.latents)

    def forward(self, x):
        x = self.maxpool(self.dconv_down1(x))
        x = self.maxpool(self.dconv_down2(x))
        conv3 = self.dconv_down3(x); x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x); x = self.maxpool(conv4)
        conv5 = self.dconv_down5(x); x = self.maxpool(conv5)
        conv6 = self.dconv_down6(x)
        x = conv6.view(conv6.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x), conv6, conv5


class SkipBlock(nn.Module):
    def __init__(self, in_filters, window):
        super(SkipBlock, self).__init__()
        self.window = window
        self.graphConv_pre = ChebConv(in_filters, 2, 1, bias=False)

    def lookup(self, pos, layer, salida=(1, 1)):
        B = pos.shape[0]; N = pos.shape[1]; h = layer.shape[-1]
        pos = pos * h
        _x1 = (self.window[0] // 2) * 1.0
        _x2 = (self.window[0] // 2 + 1) * 1.0
        _y1 = (self.window[1] // 2) * 1.0
        _y2 = (self.window[1] // 2 + 1) * 1.0
        boxes = []
        for batch in range(0, B):
            x1 = pos[batch, :, 0].reshape(-1, 1) - _x1
            x2 = pos[batch, :, 0].reshape(-1, 1) + _x2
            y1 = pos[batch, :, 1].reshape(-1, 1) - _y1
            y2 = pos[batch, :, 1].reshape(-1, 1) + _y2
            boxes.append(torch.cat([x1, y1, x2, y2], axis=1))
        skip = roi_align(layer, boxes, output_size=salida, aligned=True)
        return skip.view([B, N, -1])

    def forward(self, x, adj, conv_layer):
        pos = self.graphConv_pre(x, adj)
        skip = self.lookup(pos, conv_layer)
        return torch.cat((x, skip, pos), axis=2), pos


class GUNet(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(GUNet, self).__init__()
        self.config = config
        hw = config['inputsize'] // 32
        self.z = config['latents']
        self.encoder = EncoderConv(latents=self.z, hw=hw)
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = 1e-5
        n_nodes = config['n_nodes']
        self.filters = config['filters']
        self.K = 6
        self.window = (3, 3)
        outshape = self.filters[-1] * n_nodes[-1]
        self.dec_lin = torch.nn.Linear(self.z, outshape)
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        outsize1 = self.encoder.size[4]
        outsize2 = self.encoder.size[4]
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)
        self.SC_1 = SkipBlock(self.filters[4], self.window)
        self.graphConv_up4 = ChebConv(self.filters[4] + outsize1 + 2, self.filters[3], self.K)
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        self.SC_2 = SkipBlock(self.filters[2], self.window)
        self.graphConv_up2 = ChebConv(self.filters[2] + outsize2 + 2, self.filters[1], self.K)
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias=False)
        self.pool = Pool()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        self.mu, self.log_var, conv6, conv5 = self.encoder(x)
        z = self.sampling(self.mu, self.log_var) if self.training else self.mu
        x = F.relu(self.dec_lin(z))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        x = F.relu(self.normalization6u(self.graphConv_up6(x, self.adjacency_matrices[5]._indices())))
        x = F.relu(self.normalization5u(self.graphConv_up5(x, self.adjacency_matrices[4]._indices())))
        x, pos1 = self.SC_1(x, self.adjacency_matrices[3]._indices(), conv6)
        x = F.relu(self.normalization4u(self.graphConv_up4(x, self.adjacency_matrices[3]._indices())))
        x = self.pool(x, self.upsample_matrices[0])
        x = F.relu(self.normalization3u(self.graphConv_up3(x, self.adjacency_matrices[2]._indices())))
        x, pos2 = self.SC_2(x, self.adjacency_matrices[1]._indices(), conv5)
        x = F.relu(self.normalization2u(self.graphConv_up2(x, self.adjacency_matrices[1]._indices())))
        x = self.graphConv_up1(x, self.adjacency_matrices[0]._indices())
        return x, pos1, pos2


# ═══════════════════════════════════════════════════════════════════
# 4) LungSegmenter (원 dorga/preprocessing/segmentation.py)
#    입력 이미지(B,1,1024,1024) → 폐 마스크(B,1,1024,1024) float{0,1}
# ═══════════════════════════════════════════════════════════════════
class LungSegmenter(nn.Module):
    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model = self._build_and_load(weights_path)

    def _build_and_load(self, weights_path: str) -> GUNet:
        A, AD, D, U = genMatrixesLungsHeart()
        N1, N2 = A.shape[0], AD.shape[0]
        A = sp.csc_matrix(A).tocoo(); AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo(); U = sp.csc_matrix(U).tocoo()
        A_ = [A.copy()] * 3 + [AD.copy()] * 3
        D_ = [D.copy()]; U_ = [U.copy()]
        A_t, D_t, U_t = (
            [scipy_to_torch_sparse(x).to(self.device) for x in X] for X in (A_, D_, U_)
        )
        f = 32
        config = {
            "n_nodes": [N1, N1, N1, N2, N2, N2],
            "latents": 64,
            "inputsize": 1024,
            "filters": [2, f, f, f, f // 2, f // 2, f // 2],
            "skip_features": f,
        }
        model = GUNet(config, D_t, U_t, A_t).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        return model

    @staticmethod
    def _landmarks_to_mask(landmarks: torch.Tensor, size: int = 1024) -> torch.Tensor:
        B = landmarks.shape[0]
        masks = torch.zeros(B, size, size, dtype=torch.uint8, device=landmarks.device)
        for i in range(B):
            rl = landmarks[i, :44].reshape(-1, 1, 2).to(torch.int32).cpu().numpy()
            ll = landmarks[i, 44:94].reshape(-1, 1, 2).to(torch.int32).cpu().numpy()
            mask_np = masks[i].cpu().numpy()
            cv2.drawContours(mask_np, [rl], -1, 255, -1)
            cv2.drawContours(mask_np, [ll], -1, 255, -1)
            masks[i] = torch.from_numpy(mask_np)
        return (masks.float() / 255.0)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        landmarks = self.model(images)[0]
        landmarks = (landmarks * 1024).int()
        masks = self._landmarks_to_mask(landmarks)
        return masks.unsqueeze(1).cpu()
