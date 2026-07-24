# -*- coding: utf-8 -*-
"""A 실험(A1 식별가정 검증) 결과 집계 → Result/A/*.csv.

checkpoint/A/<arm>_s{seed}/ 의 results.json·test_preds.npz 를 읽어
δ_obs·Δg·δ_corr 보정표와 최종 판정을 CSV 로 낸다.
영역 순서 [RT, LT, RB, LB] 고정.

TODO: Experiment/A/run_all.py 의 verdict 로직을 이관해 CSV 산출로 통일.
"""
