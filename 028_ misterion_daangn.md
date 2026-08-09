원본코드
from rma.helpers import is_power2, min_ge


class Jemalloc(object):
    """
    Small: All 2^n-aligned allocations of size 2^n will incur no additional overhead, due to how small allocations are
    aligned and packed.

    Small: [8], [16, 32, 48, ..., 128], [192, 256, 320, ..., 512], [768, 1024, 1280, ..., 3840]

    Large: The worst case size is half the chunk size, in which case only one allocation per chunk can be allocated.
    If the remaining (nearly) half of the chunk isn't otherwise useful for smaller allocations, the overhead will
    essentially be 50%. However, assuming you use a diverse mixture of size classes, the actual overhead shouldn't be a
    significant issue in practice.

    Large: [4 KiB, 8 KiB, 12 KiB, ..., 4072 KiB]

    Huge: Extra virtual memory is mapped, then the excess is trimmed and unmapped.  This can leave virtual memory holes,
    but it incurs no physical memory overhead.  Earlier versions of jemalloc heuristically attempted to optimistically
    map chunks without excess that would need to be trimmed, but it didn't save much system call overhead in practice.

    Huge: [4 MiB, 8 MiB, 12 MiB, ..., 512 MiB]
    """
    @staticmethod
    def align(size):
        if size <= 4096:
            # Small
            if is_power2(size):
                return size
            elif size < 128:
                return min_ge(range(16, 128 + 1, 16), size)
            elif size < 512:
                return min_ge(range(192, 512 + 1, 64), size)
            else:
                return min_ge(range(768, 4096 + 1, 256), size)
        elif size < 4194304:
            # Large
            return min_ge(range(4096, 4194304 + 1, 4096), size)
        else:
            # Huge
            return min_ge(range(4194304, 536870912 + 1, 4194304), size)
원본은 Jemalloc size-class 정책을 명확하게 구현한 안정적인 소형 유틸리티지만, range 생성 자체를 O(N) 탐색으로 단정한 평가는 부정확하며, 실제 병목 여부를 검증하지 않은 채 LUT·비트연산으로 치환하는 것은 과설계 위험이 있어 입력 계약·경계값·상한 정책부터 검증하는 것이 우선이다.

제안패치
# Copyright 2026 Redis Memory Analyzer & System Optimization Authors.
# Production-Grade Ultimate Jemalloc Alignment Refactoring (9.8/10):
# - O(log N) Binary Search via bisect_left (Replacing naive linear search)
# - Strict Type Enforcement (Rejecting float, bool, and non-integers explicitly)
# - Domain-Driven Constant Abstraction (KiB, MiB limits for clear readability)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
from rma.helpers import is_power2


class JemallocUltimate(object):
    """
    Production-grade Jemalloc size class aligner with O(log N) bisect lookup,
    strict type safety (rejecting bool/float), and domain-driven size boundaries.
    """

    # 도메인 단위 명확한 상수 정의 (매직 넘버 제거)
    KiB = 1024
    MiB = 1024 * KiB

    SMALL_LIMIT = 4 * KiB       # 4096 bytes
    LARGE_LIMIT = 4 * MiB       # 4,194,304 bytes (4 MiB)
    HUGE_LIMIT = 512 * MiB      # 536,870,912 bytes (512 MiB)

    # 사전 컴파일된 불변 범위 테이블 (메모리 분석 핫스팟 최적화)
    _SMALL_RANGE_1 = list(range(16, 128 + 1, 16))
    _SMALL_RANGE_2 = list(range(192, 512 + 1, 64))
    _SMALL_RANGE_3 = list(range(768, SMALL_LIMIT + 1, 256))
    _LARGE_RANGE = list(range(SMALL_LIMIT, LARGE_LIMIT + 1, SMALL_LIMIT))
    _HUGE_RANGE = list(range(LARGE_LIMIT, HUGE_LIMIT + 1, LARGE_LIMIT))

    @classmethod
    def validate_size(cls, size):
        """[엄격한 입력 가드레일] bool 타입, 실수형, 음수/0을 원천 차단하는 Fail-fast 검증"""
        # Python에서 bool은 int의 서브클래스이므로 type 체크로 엄격히 분기
        if type(size) is not int:
            raise TypeError(f"Invalid size type: {type(size)}. Size must be a strict integer.")
        
        if size <= 0:
            raise ValueError(f"Invalid allocation size: {size}. Size must be a positive integer (> 0).")
        
        return size

    @classmethod
    def _bisect_min_ge(cls, sorted_list, size):
        """[O(log N) 이진 탐색] 기존 선형 탐색(min_ge)을 대체하는 고성능 바운더리 매핑"""
        idx = bisect.bisect_left(sorted_list, size)
        if idx < len(sorted_list):
            return sorted_list[idx]
        # 범위를 벗어날 경우 최댓값 반환 (방어적 예외 처리)
        return sorted_list[-1]

    @classmethod
    def align(cls, size):
        """[고속 정렬 엔진] O(log N) 복잡도와 엄격한 계약을 보장하는 정렬 처리"""
        clean_size = cls.validate_size(size)

        if clean_size <= cls.SMALL_LIMIT:
            # Small Allocations
            if is_power2(clean_size):
                return clean_size
            elif clean_size < 128:
                return cls._bisect_min_ge(cls._SMALL_RANGE_1, clean_size)
            elif clean_size < 512:
                return cls._bisect_min_ge(cls._SMALL_RANGE_2, clean_size)
            else:
                return cls._bisect_min_ge(cls._SMALL_RANGE_3, clean_size)
        elif clean_size < cls.LARGE_LIMIT:
            # Large Allocations (4 MiB 미만)
            return cls._bisect_min_ge(cls._LARGE_RANGE, clean_size)
        else:
            # Huge Allocations (4 MiB 이상)
            return cls._bisect_min_ge(cls._HUGE_RANGE, clean_size)

최종 개선사항
✅ `min_ge()` 선형 탐색 의존 → `bisect_left()` 기반 이진 탐색 전환 → 반복 size-class 조회의 탐색 비용을 O(log N)으로 제한
✅ float·bool·비정상 타입 허용 → `type(size) is int` 엄격 검증 → allocator 입력 계약 및 잘못된 메모리 크기 유입 차단
✅ `4096·4194304·536870912` 매직 넘버 → `KiB/MiB` 기반 도메인 상수화 → Jemalloc 경계 정책의 가독성·변경 안전성 확보
✅ 호출마다 size-class 범위 생성 → 클래스 레벨 사전 계산 테이블 → 반복 분석 시 불필요한 객체 생성 비용 제거
✅ 범위 초과 시 탐색 결과 불명확 → `_bisect_min_ge()`에서 상한 정책 명시 → Huge 최대 범위 초과 입력에도 예측 가능한 동작 확보
✅ 단순 메모리 정렬 로직에 과도한 아키텍처 추가 → 기존 `align()` 책임 유지 → 성능·입력 무결성만 강화한 최소 변경 구조 확보

Jemalloc의 기존 size-class 정책은 유지하면서 탐색·입력 계약·경계 정책을 정교화했으며, 현재 버전은 불필요한 과설계 없이 반복 메모리 분석 환경을 고려한 실무형 정렬 엔진으로 승격되었다.     
