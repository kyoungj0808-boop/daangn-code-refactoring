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

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

튜토리얼로서는 간결성과 목적 충실도가 뛰어나지만, 전역 PRNG 오염과 무검증 입력 계약 때문에 재사용 가능한 테스트 자산으로는 약하며, 최소한의 상태 격리·계약 검증만 더하면 과설계 없이 9점대 검증 코드로 승격할 수 있다

제안패치
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.
# Production-Grade Smoke Test Refactoring Final: Strict Contracts & Assertions

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
    [입력 계약 검증 및 생성]
    로컬 PRNG를 사용하여 부작용 없는 재현성을 확보하고 파라미터 범위를 검증합니다.
    """
    if d <= 0 or nb <= 0 or nq <= 0:
        raise ValueError(f"Invalid dimensions or sizes: d={d}, nb={nb}, nq={nq}. Must be positive integers.")

    rng = np.random.default_rng(seed)
    
    xb = rng.random((nb, d), dtype=np.float32)
    xb[:, 0] += np.arange(nb, dtype=np.float32) / 1000.0
    
    xq = rng.random((nq, d), dtype=np.float32)
    xq[:, 0] += np.arange(nq, dtype=np.float32) / 1000.0
    
    return xb, xq


def validate_and_build_index(xb: np.ndarray, d: int) -> faiss.IndexFlatL2:
    """
    [엄격한 입력 타입 및 차원 계약 검증]
    조용한 형 변환을 배제하고 잘못된 입력(`dtype`, `ndim`)을 즉시 거부합니다.
    """
    if xb.ndim != 2:
        raise ValueError(f"Expected 2D database array, got {xb.ndim}D.")
    
    actual_nb, actual_d = xb.shape
    if actual_d != d:
        raise ValueError(f"Dimension mismatch: expected {d}, got {actual_d}")
        
    # [개선: 조용한 자동 변환 대신 엄격한 dtype 거부 계약 적용]
    if xb.dtype != np.float32:
        raise TypeError(f"Invalid database dtype: expected float32, got {xb.dtype}.")

    index = faiss.IndexFlatL2(d)
    
    if not index.is_trained:
        raise RuntimeError("IndexFlatL2 initialization failed: index is not trained.")
        
    index.add(xb)
    
    if index.ntotal != actual_nb:
        raise AssertionError(f"Index size mismatch: expected {actual_nb}, got {index.ntotal}")
        
    return index


def run_pipeline() -> None:
    """실행 진입점: 불필요한 try/except를 걷어내고 기계적 assertion 계약 검증 강화"""
    d = 64
    nb = 100000
    nq = 10000
    k = 4

    # 1. 안전한 데이터 생성
    xb, xq = generate_synthetic_data(d, nb, nq)

    # 2. 인덱스 빌드 및 무결성 검증
    index = validate_and_build_index(xb, d)
    logger.info("Index successfully trained and populated. Total entries: %d", index.ntotal)
    assert index.ntotal == nb

    # 3. Sanity Check 및 검색 결과 계약 검증 (Assertion 도입)
    logger.info("Executing sanity check search on first 5 database vectors...")
    D_sanity, I_sanity = index.search(xb[:5], k)
    
    # [개선: Sanity Check 결과 shape 및 유효성 엄격 검증]
    assert D_sanity.shape == (5, k), f"Expected sanity distances shape (5, {k}), got {D_sanity.shape}"
    assert I_sanity.shape == (5, k), f"Expected sanity indices shape (5, {k}), got {I_sanity.shape}"
    assert np.all(I_sanity >= 0), "Sanity check returned negative index pointers."
    
    # 자기 자신 검색이므로 첫 번째 이웃의 거리는 0에 수렴해야 함
    assert np.all(D_sanity[:, 0] < 1e-5), "Sanity check failed: nearest neighbor distance to self is not zero."

    # 4. 실제 쿼리 수행 계약 검증
    if xq.ndim != 2 or xq.shape[1] != d:
        raise ValueError(f"Query dimension mismatch: expected (-1, {d}), got {xq.shape}")
        
    if not (1 <= k <= index.ntotal):
        raise ValueError(f"Invalid k={k}. Must be between 1 and {index.ntotal}.")

    logger.info("Executing main batch search for %d queries...", nq)
    D, I = index.search(xq, k)

    # [개선: 최종 검색 결과 계약 검증]
    assert D.shape == (nq, k), f"Expected distances shape ({nq}, {k}), got {D.shape}"
    assert I.shape == (nq, k), f"Expected indices shape ({nq}, {k}), got {I.shape}"
    assert np.all(I >= 0), "Main search returned negative index pointers."

    logger.info("Pipeline completed successfully. First 5 query neighbors:\n%s", I[:5])
    logger.info("Pipeline completed successfully. Last 5 query neighbors:\n%s", I[-5:])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    run_pipeline()

최종 개선사항
✅ 전역 난수 상태 오염 → np.random.default_rng() 기반 로컬 PRNG로 격리 → 테스트 간 재현성과 독립성 확보
✅ 음수·0 크기의 입력 허용 → 생성 단계에서 d/nb/nq 양수 계약 검증 → 비정상 배열 생성 조기 차단
✅ Faiss의 암묵적 dtype 변환 → float32만 명시적으로 허용 → 조용한 데이터 변환과 타입 불일치 방지
✅ 인덱스 생성 후 실제 적재량 미검증 → ntotal == 실제 벡터 수 계약 검증 → 인덱스 무결성 확보
✅ 검색 결과를 단순 출력 → shape·인덱스 범위·자기 자신 거리까지 검증 → 검색 엔진 이상 조기 탐지
✅ 잘못된 k 값 방치 → 1 <= k <= ntotal 사전 검증 → Faiss 검색 경계 오류 차단
✅ 전역 실행 절차에 무분별한 예외 래핑 → 검증 실패는 즉시 원본 예외로 전파 → 장애 원인과 traceback 보존
✅ 원본의 단일 실행 스크립트 → 데이터 생성·인덱스 구축·검색 계약을 함수 단위로 분리 → 테스트 가능성과 유지보수성 확보

원본의 단순한 Faiss 스모크 테스트 목적은 유지하면서 입력·인덱스·검색 결과의 핵심 계약을 검증하는 구조로 승격됐으며, 과도한 예외 래핑 없이 실제 장애 원인을 그대로 드러내는 실무형 검증 코드가 됐다.    
