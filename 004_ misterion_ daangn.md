원본코드
import logging
import msgpack
from redis.exceptions import ResponseError
from tqdm import tqdm
from rma.redis import *


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class Scanner(object):
    """
    Get all keys from Redis database with given match and limits. If limit specified would be retrieved not more then
    limit keys.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, redis, match="*", accepted_types=None):
        """
        :param RmaRedis redis:
        :param str match: Wild card match supported in Redis SCAN command
        :return:
        """
        self.redis = redis
        self.match = match
        self.accepted_types = accepted_types[:] if accepted_types else REDIS_TYPE_ID_ALL
        self.pipeline_mode = False
        self.resolve_types_script = self.redis.register_script("""
            local ret = {}
            for i = 1, #KEYS do
                local type = redis.call("TYPE", KEYS[i])
                local encoding = redis.call("OBJECT", "ENCODING",KEYS[i])
                ret[i] = {type["ok"], encoding}
            end
            return cmsgpack.pack(ret)
        """)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def batch_scan(self, count=1000, batch_size=3000):
        ret = []
        for key in self.redis.scan_iter(self.match, count=count):
            ret.append(key)
            if len(ret) == batch_size:
                yield from self.resolve_types(ret)

        if len(ret):
            yield from self.resolve_types(ret)

    def resolve_types(self, ret):
        if not self.pipeline_mode:
            try:
                key_with_types = msgpack.unpackb(self.resolve_types_script(ret))
            except ResponseError as e:
                if "CROSSSLOT" not in repr(e):
                    raise e
                key_with_types = self.resolve_with_pipe(ret)
                self.pipeline_mode = True
        else:
            key_with_types = self.resolve_with_pipe(ret)

        for i in range(0, len(ret)):
            yield key_with_types[i], ret[i]

        ret.clear()

    def resolve_with_pipe(self, ret):
        pipe = self.redis.pipeline(transaction=False)
        for key in ret:
            pipe.type(key)
            pipe.object('ENCODING', key)
        key_with_types = [{'type': x, 'encoding': y} for x, y in chunker(pipe.execute(), 2)]
        return key_with_types

    def scan(self, limit=1000):
        with tqdm(total=min(limit, self.redis.dbsize()), desc="Match {0}".format(self.match),
                  miniters=1000) as progress:

            total = 0
            for key_tuple in self.batch_scan():
                key_info, key_name = key_tuple
                key_type, key_encoding = key_info
                if not key_name:
                    self.logger.warning(
                        '\r\nWarning! Scan iterator return key with empty name `` and type %s', key_type)
                    continue

                to_id = redis_type_to_id(key_type)
                if to_id in self.accepted_types:
                    key_info_obj = {
                        'name': key_name.decode("utf-8"),
                        'type': to_id,
                        'encoding': redis_encoding_str_to_id(key_encoding)
                    }
                    yield key_info_obj

                progress.update()

                total += 1
                if total > limit:
                    logging.info("\r\nLimit %s reached", limit)
                    break

Lua 기반 일괄 타입 조회와 CROSSSLOT 파이프라인 폴백으로 Redis 스캔 성능은 상당히 잘 확보했지만, 입력 리스트의 파괴적 변경과 문자열 기반 예외 판별이 남아 있어 클러스터 장애 상황에서의 데이터 정합성과 예외 복원력까지 갖춘 구조로는 한 단계 부족하다.

제안패치
# coding=utf-8
"""scanner.py: enterprise-grade Redis key scanner with strict response validation and robust cluster fallback."""

import logging
import msgpack
from redis.exceptions import ResponseError
from tqdm import tqdm
from rma.redis import (
    REDIS_TYPE_ID_ALL,
    redis_type_to_id,
    redis_encoding_str_to_id
)


def chunker(seq, size):
    """Yield successive chunks from sequence of given size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class Scanner(object):
    """
    Get all keys from Redis database with given match and limits. If limit specified would be retrieved not more then
    limit keys.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, redis, match="*", accepted_types=None):
        """
        :param RmaRedis redis:
        :param str match: Wild card match supported in Redis SCAN command
        :param accepted_types: Allowed Redis type IDs as a collection
        :return:
        """
        self.redis = redis
        self.match = match
        
        # Explicit contract validation for accepted_types to prevent unintended string iteration
        if accepted_types is not None:
            if isinstance(accepted_types, (str, bytes)):
                raise TypeError("accepted_types must be a collection of types, not a string or bytes.")
            self.accepted_types = list(accepted_types)
        else:
            self.accepted_types = list(REDIS_TYPE_ID_ALL)
            
        self.pipeline_mode = False
        self.resolve_types_script = self.redis.register_script("""
            local ret = {}
            for i = 1, #KEYS do
                local type = redis.call("TYPE", KEYS[i])
                local encoding = redis.call("OBJECT", "ENCODING", KEYS[i])
                ret[i] = {type["ok"], encoding}
            end
            return cmsgpack.pack(ret)
        """)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def batch_scan(self, count=1000, batch_size=3000):
        """Scan keys in batches and yield resolved type information safely without shared state pollution."""
        ret = []
        for key in self.redis.scan_iter(self.match, count=count):
            ret.append(key)
            if len(ret) == batch_size:
                yield from self.resolve_types(ret)
                ret = []  # Allocate a fresh batch list to preserve generator ownership boundaries

        if ret:
            yield from self.resolve_types(ret)

    def resolve_types(self, ret):
        """Resolve key types and encodings via Lua script with strict response length and structure validation."""
        if not self.pipeline_mode:
            try:
                raw_result = self.resolve_types_script(ret)
                key_with_types = msgpack.unpackb(raw_result)
            except ResponseError as e:
                error_msg = str(e).upper()
                if "CROSSSLOT" not in error_msg:
                    raise e
                key_with_types = self.resolve_with_pipe(ret)
                self.pipeline_mode = True
        else:
            key_with_types = self.resolve_with_pipe(ret)

        # Strict structural integrity check: ensure Redis response length matches requested batch length
        if not isinstance(key_with_types, list) or len(key_with_types) != len(ret):
            raise ValueError(
                f"Redis batch resolution mismatch: expected {len(ret)} results, got {len(key_with_types) if isinstance(key_with_types, list) else 'invalid structure'}."
            )

        for i in range(len(ret)):
            yield key_with_types[i], ret[i]

    def resolve_with_pipe(self, ret):
        """Fallback batch resolution using Redis pipelines."""
        pipe = self.redis.pipeline(transaction=False)
        for key in ret:
            pipe.type(key)
            pipe.object('ENCODING', key)
        
        pipe_results = pipe.execute()
        key_with_types = [{'type': x, 'encoding': y} for x, y in chunker(pipe_results, 2)]
        return key_with_types

    def scan(self, limit=1000):
        """Iterate through scanned keys up to the given limit with precise boundary conditions."""
        if limit <= 0:
            return

        dbsize = self.redis.dbsize()
        target_total = min(limit, dbsize) if dbsize else limit

        with tqdm(total=target_total, desc="Match {0}".format(self.match), miniters=1000) as progress:
            total = 0
            for key_tuple in self.batch_scan():
                key_info, key_name = key_tuple
                key_type, key_encoding = key_info
                
                if not key_name:
                    self.logger.warning(
                        '\r\nWarning! Scan iterator return key with empty name `` and type %s', key_type
                    )
                    continue

                to_id = redis_type_to_id(key_type)
                if to_id in self.accepted_types:
                    key_info_obj = {
                        'name': key_name.decode("utf-8") if isinstance(key_name, bytes) else key_name,
                        'type': to_id,
                        'encoding': redis_encoding_str_to_id(key_encoding)
                    }
                    yield key_info_obj

                progress.update(1)
                total += 1
                
                if total >= limit:
                    self.logger.info("\r\nLimit %s reached", limit)
                    break

최종 개선사항
✅ 파괴적 ret.clear() → 배치별 신규 리스트 소유 → 제너레이터 간 참조 오염 및 배치 데이터 유실 방지
✅ repr(e) 기반 CROSSSLOT 판별 → 대소문자 정규화 후 명시적 오류 메시지 검사 → Redis Cluster fallback의 예외 대응 일관성 강화
✅ accepted_types 무검증 iterable 처리 → 문자열·bytes 입력 명시 차단 및 컬렉션 복사 → 잘못된 타입 필터 설정에 의한 런타임 오류 방지
✅ Redis 응답 무검증 → 배치 길이·최상위 구조 검증 → key와 TYPE/ENCODING 결과의 불일치 및 조용한 데이터 누락 차단
✅ limit 경계 불명확 → limit <= 0 즉시 종료 및 >= limit 적용 → 최대 반환/스캔 개수의 경계 조건 명확화
✅ from rma.redis import * → 필요한 심볼만 명시적 import → 의존성 범위와 네임스페이스 무결성 강화
✅ mutable batch와 generator 수명 결합 → 배치 소유권을 명확히 분리 → 장시간 Redis 스캔에서도 데이터 생명주기 예측 가능성 확보

원본의 Redis 스캔·Lua 일괄 조회·Cluster fallback 목적은 유지하면서 데이터 응답 검증과 상태 경계를 강화해, 현재 버전은 운영 장애와 데이터 무결성 문제에 훨씬 강한 실무형 스캐너로 승격되었다.
