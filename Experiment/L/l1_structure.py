# -*- coding: utf-8 -*-
r"""L0–L1 — 구조·분포 진단 → Result/L/l1_structure.csv  (draft §4.2)

증거 사다리 최하단. labels.csv 만으로 계산 가능(모델·영상 불필요).

계산 항목:
  · L0 무결성   : image-label 매칭 불일치 수, 환자 단위 split 누수 수 (둘 다 0 기대)
  · L1 분포     : 환자수/영상수, 환자당 시퀀스 길이(연도간 Mann-Whitney),
                  전체 평균등급, 등급 0·4 비율
  · 마르코프    : 환자별 1차 등급 전이행렬의 방향비(악화/유지/호전) 연도간 비교
                  (질병 동역학 동일성 확인 → 차이는 동역학이 아니라 출발 분포·기준)

영역 순서 [RT, LT, RB, LB] 고정. 연도 = patient_id 접두어 24_/26_.

TODO: labels.csv 로드 → 위 지표 표(표1) 산출 → Result/L/l1_structure.csv.
"""
