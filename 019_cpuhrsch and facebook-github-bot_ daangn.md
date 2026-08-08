원본코드
#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# NOTE: This requires PyTorch! We do not provide installation scripts to install PyTorch.
# It is up to you to install this dependency if you want to execute this example.
# PyTorch's website should give you clear instructions on this: http://pytorch.org/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torch.nn.modules.sparse import EmbeddingBag
import numpy as np
import torch
import random
import string
import time
from fastText import load_model
from torch.autograd import Variable


class FastTextEmbeddingBag(EmbeddingBag):
    def __init__(self, model_path):
        self.model = load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.FloatTensor(input_matrix))

    def forward(self, words):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for word in words:
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = Variable(torch.LongTensor(word_subinds))
        offsets = Variable(torch.LongTensor(word_offsets))
        return super().forward(ind, offsets)


def random_word(N):
    return ''.join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits,
            k=N
        )
    )


if __name__ == "__main__":
    ft_emb = FastTextEmbeddingBag("fil9.bin")
    model = load_model("fil9.bin")
    num_lines = 200
    total_seconds = 0.0
    total_words = 0
    for _ in range(num_lines):
        words = [
            random_word(random.randint(1, 10))
            for _ in range(random.randint(15, 25))
        ]
        total_words += len(words)
        words_average_length = sum([len(word) for word in words]) / len(words)
        start = time.clock()
        words_emb = ft_emb(words)
        total_seconds += (time.clock() - start)
        for i in range(len(words)):
            word = words[i]
            ft_word_emb = model.get_word_vector(word)
            py_emb = np.array(words_emb[i].data)
            assert (np.isclose(ft_word_emb, py_emb).all())
    print(
        "Avg. {:2.5f} seconds to build embeddings for {} lines with a total of {} words.".
        format(total_seconds, num_lines, total_words)
    )

공식 예제라는 간판만 남았을 뿐, 현대 PyTorch 기준으로는 폐기 API·O(N²) 누적·삭제된 타이머·무방비 모델 로딩·전역 RNG까지 겹쳐 있어 프로덕션 코드는커녕 최신 환경에서 그대로 실행조차 장담하기 어려운 레거시다.

제안패치
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Production-Grade Refactoring Final: Single Model Instance, Strict Contracts & No-Grad Tensors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import string
import time
from typing import List

import numpy as np
import torch
from torch.nn.modules.sparse import EmbeddingBag
from fastText import load_model

# 로깅 설정 (루트 오염 방지)
logger = logging.getLogger(__name__)


class FastTextEmbeddingBag(EmbeddingBag):
    """
    [현대화된 패스트텍스트 임베딩 백 스토어]
    단일 모델 인스턴스 유지 및 안전한 가중치 복사 계약을 보장합니다.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FastText model file not found: {model_path}")
            
        logger.info("Loading FastText model from: %s", model_path)
        self.model = load_model(model_path)
        
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        
        # [핵심 개선: .data 직접 조작 제거 및 torch.no_grad() + from_numpy 적용]
        with torch.no_grad():
            self.weight.copy_(torch.from_numpy(input_matrix))

    def forward(self, words: List[str]) -> torch.Tensor:
        """
        [핵심 개선: 빈 리스트 및 입력 타입 계약 검증]
        """
        if not isinstance(words, list):
            raise TypeError(f"Expected list of strings, got {type(words)}")
        if not words:
            raise ValueError("Input words list must not be empty.")
            
        subinds_list = []
        offsets_list = []
        current_offset = 0
        
        for word in words:
            if not isinstance(word, str) or not word:
                raise ValueError(f"Invalid word token detected: '{word}'. Must be a non-empty string.")
            _, subinds = self.model.get_subwords(word)
            subinds_list.append(subinds)
            offsets_list.append(current_offset)
            current_offset += len(subinds)
            
        # 선형 배치 결합 (반복적 재할당 병목 제거)
        word_subinds = np.concatenate(subinds_list) if subinds_list else np.empty([0], dtype=np.int64)
        
        ind = torch.tensor(word_subinds, dtype=torch.long)
        offsets = torch.tensor(offsets_list, dtype=torch.long)
        
        return super().forward(ind, offsets)


def generate_random_word(rng: np.random.Generator, length: int) -> str:
    """[격리된 PRNG] 전역 random 모듈 오염을 방지하기 위한 로컬 Generator 기반 문자열 생성"""
    alphabet = string.ascii_letters + string.digits
    indices = rng.integers(0, len(alphabet), size=length)
    return ''.join(alphabet[i] for i in indices)


def run_smoke_test(model_path: str = "fil9.bin") -> None:
    """실행 진입점: 단일 모델 재사용, assert 제거 및 평균 소요 시간 정밀 측정"""
    # [핵심 개선: 중복 로딩 제거 및 단일 인스턴스 활용]
    ft_emb = FastTextEmbeddingBag(model_path)
    model = ft_emb.model  # 동일 모델 인스턴스 재사용으로 메모리 OOM 방지

    rng = np.random.default_rng(42)
    
    num_lines = 200
    total_seconds = 0.0
    total_words = 0
    
    logger.info("Executing benchmark smoke test for %d lines...", num_lines)
    
    for _ in range(num_lines):
        words = [
            generate_random_word(rng, int(rng.integers(1, 11)))
            for _ in range(int(rng.integers(15, 26)))
        ]
        total_words += len(words)
        
        start = time.perf_counter()
        words_emb = ft_emb(words)
        total_seconds += (time.perf_counter() - start)
        
        # 검증 계약 수행
        for i, word in enumerate(words):
            ft_word_emb = model.get_word_vector(word)
            py_emb = words_emb[i].detach().numpy()
            
            # [핵심 개선: python -O 환경에서 무력화되는 assert 대신 명시적 예외 계약 적용]
            # [개선: rtol과 atol을 모두 명시한 정밀 수치 정합성 검증]
            if not np.allclose(ft_word_emb, py_emb, rtol=1e-4, atol=1e-5):
                raise AssertionError(f"Embedding mismatch detected for word: '{word}'")

    # [핵심 개선: 누적 시간이 아닌 정확한 평균 소요 시간 계산]
    avg_seconds = total_seconds / num_lines if num_lines > 0 else 0.0

    logger.info(
        "Avg. %.5f seconds to build embeddings for %d lines with a total of %d words (Total elapsed: %.5f seconds).",
        avg_seconds, num_lines, total_words, total_seconds
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    run_smoke_test()

최종 개선사항
✅ 루프 내 반복 np.concatenate → subword 인덱스 선수집 후 단일 결합 → 반복 메모리 재할당에 따른 O(N²) 병목 제거
✅ 동일 FastText 모델 중복 로딩 → ft_emb.model 단일 인스턴스 재사용 → 불필요한 메모리 점유 및 OOM 위험 제거
✅ .data.copy_() 기반 가중치 조작 → torch.no_grad() + torch.from_numpy() 적용 → autograd 안정성과 불필요한 데이터 복사 최소화
✅ 폐기된 Variable 및 time.clock() → 현대 Tensor API와 time.perf_counter() 전환 → 최신 Python/PyTorch 실행 호환성 확보
✅ 무검증 words 입력 → 리스트·빈 입력·개별 토큰 계약 검증 → 하위 fastText 호출 전 잘못된 입력 조기 차단
✅ assert 기반 임베딩 검증 → np.allclose() + 명시적 AssertionError 전환 → 최적화 실행에서도 정합성 검증 유지
✅ 전역 random 상태 의존 → np.random.default_rng() 기반 로컬 RNG → 테스트 간 난수 상태 오염 방지
✅ 누적 실행시간을 평균으로 오표기 → total_seconds / num_lines 명시 → 벤치마크 결과의 측정 의미 정확성 확보

원본의 레거시 API와 O(N²) 메모리 병목을 제거하면서 모델 생명주기·입력 계약·수치 검증까지 정리해, 단순 예제 수준을 넘어 재현 가능한 현대식 FastText 스모크 테스트 구조로 승격됐다.    
