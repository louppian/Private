#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stn.py — Spatial Transformer Network (폐 마스크 기반 affine 정렬)

원본: C:\\Code\\DORGA\\dorga\\models\\stn.py 를 외부 의존 없이 이식.
가중치: stn_weights.pth (= CXR_GUI_V4/assets/weights/stn_weights.pth)

의존: torch 만.  state_dict 호환을 위해 레이어 이름/구조는 원본과 동일하게 유지한다.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


# 항등 affine (정렬 불필요 시 게이트로 되돌릴 기준)
IDENTITY_THETA = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0]], dtype=torch.float32
)


class STN(nn.Module):
    """마스크(1,512,512) → 2x3 affine theta 예측 → 이미지에 grid_sample."""

    def __init__(self, in_shape=(1, 512, 512), mask_resize: int = 512,
                 dense_neurons=50, freeze_align_model=False,
                 identity_gate_threshold=0.01):
        super(STN, self).__init__()

        assert not in_shape[1] % mask_resize, "STN size must be a multiple of mask size"
        trainable = not freeze_align_model
        self.identity_gate_threshold = identity_gate_threshold

        self.pool1 = nn.MaxPool2d(
            kernel_size=(in_shape[1] // mask_resize, in_shape[2] // mask_resize)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv1.requires_grad_(False)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1)
        if not trainable:
            self.conv2.requires_grad_(False)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.calculate_flatten_size(in_shape),
                             out_features=dense_neurons)
        if not trainable:
            self.fc1.requires_grad_(False)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=dense_neurons, out_features=6)
        if not trainable:
            self.fc2.requires_grad_(False)

        # affine 회귀 헤드는 항등변환으로 초기화
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def calculate_flatten_size(self, in_shape):
        dummy = torch.randn(1, in_shape[0], in_shape[1], in_shape[2])
        x = self.pool1(dummy)
        x = self.pool2(self.conv1(x))
        x = self.pool3(self.conv2(x))
        return x.numel()

    def predict_theta(self, x):
        xs = self.pool1(x)
        xs = self.pool2(self.conv1(xs))
        xs = self.pool3(self.conv2(xs))
        xs = self.flatten(xs)
        xs = self.relu(self.fc1(xs))
        theta = self.fc2(xs)
        return theta.view(-1, 2, 3)

    @staticmethod
    def transform(x, theta):
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, x):
        theta = self.predict_theta(x)
        if not self.training and self.identity_gate_threshold > 0:
            theta = self._apply_identity_gate(theta)
        return theta

    def _apply_identity_gate(self, theta):
        """예측 theta가 항등에 충분히 가까우면 항등으로 스냅(과도 정렬 방지)."""
        identity = IDENTITY_THETA.to(theta.device).unsqueeze(0)
        diff = (theta - identity).abs().mean(dim=(1, 2))
        mask = (diff < self.identity_gate_threshold).float().view(-1, 1, 1)
        return mask * identity + (1 - mask) * theta