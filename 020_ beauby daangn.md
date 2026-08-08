원본코드
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss

nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])

튜토리얼로서는 IVFPQ의 핵심 흐름을 정확히 보여주지만, 전역 RNG·PQ 파라미터·학습 데이터 규모·is_trained 계약을 검증하지 않아 운영 환경에서는 설정 오류가 인덱스 구축 장애로 직결될 수 있는 구조다.

제안패치
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.
# Production-Grade Refactoring V3: Strict Parameter Ranges, nprobe Bounds & Sentinel-Aware Validation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from typing import Tuple
import numpy as np
import faiss

# 모듈별 로거 구성 (루트 로깅 오염 방지)
logger = logging.getLogger(__name__)


def generate_synthetic_data(d: int, nb: int, nq: int, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray]:
    """
    [격리된 PRNG 데이터 생성]
    전역 상태 오염을 방지하고 입력 차원 및 크기의 유효성을 사전에 검증합니다.
    """
    if d <= 0 or nb <= 0 or nq <= 0:
        raise ValueError(f"Invalid dimensions or sizes: d={d}, nb={nb}, nq={nq}. Must be positive integers.")

    logger.info("Generating synthetic vectors with isolated RNG (dim=%d, database=%d, queries=%d)", d, nb, nq)
    rng = np.random.default_rng(seed)
    
    xb = rng.random((nb, d), dtype=np.float32)
    xb[:, 0] += np.arange(nb, dtype=np.float32) / 1000.0
    
    xq = rng.random((nq, d), dtype=np.float32)
    xq[:, 0] += np.arange(nq, dtype=np.float32) / 1000.0
    
    return xb, xq


def validate_and_build_ivfpq_index(
    xb: np.ndarray, 
    d: int, 
    nlist: int, 
    m: int, 
    nbits: int = 8
) -> faiss.IndexIVFPQ:
    """
    [엄격한 파라미터 개별 범위 계약 및 수학적 제원 검증]
    - 파라미터 양수 및 비정상 값(m=0 등) 원천 차단
    - d가 m의 배수인지 검증 (d % m == 0)
    - 훈련 및 인덱스 적재 무결성 검증
    """
    # [개선: 개별 파라미터의 기초 범위 계약 검증 (m=0 ZeroDivisionError 방어 포함)]
    if m <= 0:
        raise ValueError(f"Number of sub-vectors m={m} must be greater than 0.")
    if nlist <= 0:
        raise ValueError(f"Number of clusters nlist={nlist} must be greater than 0.")
    if nbits <= 0:
        raise ValueError(f"Number of bits nbits={nbits} must be greater than 0.")

    if xb.ndim != 2:
        raise ValueError(f"Expected 2D database array, got {xb.ndim}D.")
    
    actual_nb, actual_d = xb.shape
    if actual_d != d:
        raise ValueError(f"Dimension mismatch: expected {d}, got {actual_d}")
        
    if xb.dtype != np.float32:
        raise TypeError(f"Invalid database dtype: expected float32, got {xb.dtype}.")

    # [수학적 제원 계약 검증: d는 반드시 m의 배수여야 함]
    if d % m != 0:
        raise ValueError(f"Dimension d={d} must be divisible by number of sub-vectors m={m}.")

    logger.info("Initializing IndexIVFPQ quantizer and index (d=%d, nlist=%d, m=%d, nbits=%d)", d, nlist, m, nbits)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # [개선: 불필요한 초기 is_trained 검증 제거 후 실제 학습 진입]
    logger.info("Training IndexIVFPQ index with %d database vectors...", actual_nb)
    index.train(xb)

    # [인덱스 학습 상태 명시적 계약 검증 (python -O 무관한 예외 전환)]
    if not index.is_trained:
        raise RuntimeError("IndexIVFPQ training failed: index remains untrained.")

    logger.info("Adding vectors to trained IndexIVFPQ index...")
    index.add(xb)

    # [적재 데이터 크기 정합성 검증]
    if index.ntotal != actual_nb:
        raise AssertionError(f"Index size mismatch: expected {actual_nb}, got {index.ntotal}")

    return index


def run_pipeline() -> None:
    """실행 진입점: IVFPQ 학습, Sanity Check, nprobe 계약 검증 및 sentinel 대응 배치 검색"""
    d = 64
    nb = 100000
    nq = 10000
    nlist = 100
    m = 8
    k = 4

    # 1. 안전한 데이터 생성
    xb, xq = generate_synthetic_data(d, nb, nq)

    # 2. IVFPQ 인덱스 빌드, 학습 및 무결성 검증
    index = validate_and_build_ivfpq_index(xb, d, nlist, m, nbits=8)
    logger.info("IndexIVFPQ successfully trained, populated, and verified. Total entries: %d", index.ntotal)

    # 3. Sanity Check 및 검색 결과 계약 검증 (센티넬 sentinel 허용)
    logger.info("Executing sanity check search on first 5 database vectors...")
    D_sanity, I_sanity = index.search(xb[:5], k)
    
    if D_sanity.shape != (5, k) or I_sanity.shape != (5, k):
        raise AssertionError(f"Invalid sanity search shape: D={D_sanity.shape}, I={I_sanity.shape}")
    
    # [개선: Faiss 누락 결과 sentinel(-1)은 허용하되, 비정상 음수(-2 이하)는 차단]
    if np.any(I_sanity < -1):
        raise AssertionError("Sanity check returned invalid negative index pointers (below -1 sentinel).")

    logger.debug("Sanity Check Indices:\n%s", I_sanity)
    logger.debug("Sanity Check Distances:\n%s", D_sanity)

    # 4. nprobe 설정 및 범위 계약 검증 후 실제 쿼리 수행
    nprobe = 10
    
    # [개선: nprobe 실무형 범위 계약 검증 (1 <= nprobe <= nlist)]
    if not (1 <= nprobe <= nlist):
        raise ValueError(f"Invalid nprobe={nprobe}. Must be between 1 and nlist ({nlist}).")

    index.nprobe = nprobe
    logger.info("Configured nprobe to %d within valid bounds. Executing main batch search...", nprobe)

    if xq.ndim != 2 or xq.shape[1] != d:
        raise ValueError(f"Query dimension mismatch: expected (-1, {d}), got {xq.shape}")

    D, I = index.search(xq, k)

    # 최종 검색 결과 계약 검증 (센티넬 대응)
    if D.shape != (nq, k) or I.shape != (nq, k):
        raise AssertionError(f"Invalid search output shape: D={D.shape}, I={I.shape}")
    if np.any(I < -1):
        raise AssertionError("Main search returned invalid negative index pointers (below -1 sentinel).")

    logger.info("Pipeline completed successfully. Last 5 query neighbors:\n%s", I[-5:])
    logger.info("Pipeline completed successfully. Last 5 query distances:\n%s", D[-5:])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    run_pipeline()

최종 개선사항
✅ 전역 난수 상태 공유 → 로컬 np.random.default_rng() 격리 → 재현성과 모듈 간 실행 독립성 확보
✅ 파라미터 범위 무검증 → m/nlist/nbits 개별 계약 검증 → ZeroDivisionError 및 비정상 Faiss 설정 사전 차단
✅ d % m 및 배열 형상·dtype 미검증 → IVFPQ 수학적·입력 데이터 계약 검증 → C++ 바인딩 단계의 모호한 실패 방지
✅ 학습 완료 여부 및 적재량 무검증 → is_trained·ntotal 명시적 검증 → 인덱스 상태와 데이터 무결성 확보
✅ 검색 결과 음수 전면 거부 → Faiss의 -1 sentinel 허용 및 -2 이하 차단 → 정상적인 미매칭 결과와 실제 오류 구분
✅ 무검증 nprobe 설정 → 1 <= nprobe <= nlist 범위 계약 적용 → 검색 비용과 검색 파라미터 오설정 방지
✅ assert 의존 검증 → 명시적 예외 계약으로 전환 → python -O에서도 핵심 무결성 검증 유지

튜토리얼 수준의 단순 IVFPQ 실행 코드를 입력·학습·적재·검색 전 단계의 계약을 검증하는 방어적 벡터 검색 파이프라인으로 승격했으며, 현재 버전은 장애 방어와 데이터 무결성을 강화하면서도 원래의 Faiss 학습·검색 목적을 과도하게 훼손하지 않은 실무형 구조다.
