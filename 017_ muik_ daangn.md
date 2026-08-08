원본코드from collections import Counter

import pandas as pd

MIN_COUNT = 10

def username_char_dict():
    df = pd.read_csv('data/emb.csv')

    chars_list = df['user_name'].apply(lambda x: list(str(x).decode('utf-8')))
    chars = [char for word in chars_list for char in word]

    print('total chars count', len(set(chars)))

    c = Counter(chars)
    write_char_dict(c.most_common(), 'data/char_dict.txt', MIN_COUNT)

def write_char_dict(most_common, filepath, min_count):
    with open(filepath, 'w') as f:
        for char, count in most_common:
            if count < min_count:
                break
            f.write("%s\n" % char.encode('utf-8'))

def get_chars(filepath):
    chars = []
    with open(filepath, 'r') as f:
        for line in f:
            chars += list(line.decode('utf-8').rstrip().replace(' ', ''))
    return chars

def text_char_dict():
    filepath = 'data/title_normalized.txt.emb.words'
    title_chars = get_chars(filepath)

    filepath = 'data/content_normalized.txt.emb.words'
    content_chars = get_chars(filepath)

    c = Counter(title_chars + content_chars)
    write_char_dict(c.most_common(), 'data/text_char_dict.txt', MIN_COUNT)

if __name__ == '__main__':
    username_char_dict()
    text_char_dict()

Python 2 레거시 전처리라는 전제에서는 목적에 충실하지만, Python 3 환경에서는 .decode()에서 즉시 깨지고 결측치·파일 I/O·인코딩 계약도 보장되지 않아 운영 파이프라인으로는 6.5점 수준이다.

제안패치
# Copyright (c) Daangn Market. All rights reserved.
# Production-Grade Refactoring V2: Strict UTF-8, Streaming Counter & Atomic File Write
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import logging
import os
import tempfile
from typing import List, Tuple

import pandas as pd

# 로깅 설정 (모듈별 가시성 확보)
logger = logging.getLogger(__name__)

MIN_COUNT = 10
DATA_DIR = 'data'


def validate_file_exists(filepath: str) -> None:
    """[방어적 I/O] 대상 파일의 존재 여부를 사전에 검증합니다."""
    if not os.path.exists(filepath):
        logger.error("Required data file missing: %s", filepath)
        raise FileNotFoundError(f"Target file does not exist: {filepath}")


def write_char_dict(most_common: List[Tuple[str, int]], filepath: str, min_count: int) -> None:
    """
    [핵심 개선: 원자적 파일 쓰기(Atomic Write) 보장]
    직접 파일 truncate를 방지하기 위해 임시 파일에 먼저 기록 후 
    fsync 및 os.replace를 통해 기존 사전을 안전하게 교체합니다.
    """
    parent = os.path.dirname(filepath)
    if parent:
        # [핵심 개선: 빈 디렉터리 문자열('') 방어]
        os.makedirs(parent, exist_ok=True)
        
    logger.info("Writing filtered character dictionary to: %s (min_count=%d)", filepath, min_count)
    
    # 디렉터리 경로 확보 및 안전한 임시 파일 생성
    temp_fd, temp_path = tempfile.mkstemp(dir=parent if parent else '.', text=True)
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            for char, count in most_common:
                if count < min_count:
                    break
                f.write(f"{char}\n")
            # [안정성 강화: 디스크 플러시 강제 동기화]
            f.flush()
            os.fsync(f.fileno())
            
        # 원자적 파일 교체 (Atomic Replace)
        os.replace(temp_path, filepath)
    except Exception:
        # 실패 시 임시 파일 정리
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def username_char_dict(csv_path: str = os.path.join(DATA_DIR, 'emb.csv'), 
                       output_path: str = os.path.join(DATA_DIR, 'char_dict.txt')) -> None:
    """
    [핵심 개선: 스트리밍 집계 및 불필요한 예외 래핑 제거]
    거대한 리스트를 메모리에 누적하지 않고 Counter.update로 스트리밍 처리합니다.
    """
    validate_file_exists(csv_path)
    logger.info("Processing user names from: %s", csv_path)
    
    df = pd.read_csv(csv_path)
    if 'user_name' not in df.columns:
        raise KeyError("Column 'user_name' not found in CSV file.")
    
    valid_names = df['user_name'].dropna().astype(str)
    
    # [핵심 개선: 메모리 절감 및 리스트 컨프리헨션 제거]
    c = Counter()
    for name in valid_names:
        c.update(name)
        
    logger.info("Total unique characters found in usernames: %d", len(c))
    write_char_dict(c.most_common(), output_path, MIN_COUNT)


def update_counter_from_file(filepath: str, counter: Counter) -> None:
    """
    [핵심 개선: Strict UTF-8 및 스트리밍 메모리 최적화]
    errors='ignore'를 완전히 제거하여 손상된 인코딩을 조용히 숨기지 않고 
    엄격하게 예외를 발생시키며, 대규모 리스트 생성 없이 Counter에 직접 적재합니다.
    """
    validate_file_exists(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:  # strict 모드 기본 적용 (errors='ignore' 제거)
        for line in f:
            cleaned_line = line.rstrip().replace(' ', '')
            counter.update(cleaned_line)


def text_char_dict() -> None:
    """타이틀 및 본문 텍스트 데이터셋을 스트리밍 방식으로 취합하여 공통 문자 사전 구축"""
    title_path = os.path.join(DATA_DIR, 'title_normalized.txt.emb.words')
    content_path = os.path.join(DATA_DIR, 'content_normalized.txt.emb.words')
    output_path = os.path.join(DATA_DIR, 'text_char_dict.txt')

    logger.info("Loading text character files via streaming pipeline...")
    c = Counter()
    
    update_counter_from_file(title_path, c)
    update_counter_from_file(content_path, c)

    logger.info("Total combined text unique characters: %d", len(c))
    write_char_dict(c.most_common(), output_path, MIN_COUNT)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    username_char_dict()
    text_char_dict()

최종 개선사항
✅ 전체 문자 리스트 누적 → Counter 직접 스트리밍 집계 → 대규모 입력에서도 메모리 사용량 최소화
✅ errors=ignore 기반 조용한 문자 손실 → strict UTF-8 처리 → 인코딩 손상을 숨기지 않고 즉시 탐지
✅ 기존 파일 직접 truncate → 임시 파일 기록 후 fsync·os.replace → 생성 중 장애에도 기존 정상 사전 보존
✅ 출력 경로가 없는 경우 무조건 makedirs → parent 경로 존재 여부 확인 후 조건부 생성 → 상대경로 입력까지 안전하게 지원
✅ 입력 파일 부재를 런타임 I/O에서 발견 → 사전 경로 검증 → 데이터 파이프라인 실패 지점을 조기에 명확화
✅ 예외를 무차별 로깅 후 재래핑 → 원본 예외와 traceback 유지 → 장애 원인 은폐 및 디버깅 정보 손실 방지
✅ 타이틀·본문을 별도 리스트로 합산 → 하나의 Counter에 순차 누적 → 중간 데이터 복제와 불필요한 메모리 사용 제거
✅ 파일 쓰기 실패 시 임시 파일 잔존 → 실패 시 임시 산출물 정리 → 불완전한 중간 파일의 운영 환경 잔류 방지

원본의 단순한 문자 사전 생성 목적은 그대로 유지하면서 메모리 효율·인코딩 무결성·파일 교체 원자성까지 확보해, 불필요한 과설계 없이 실제 데이터 파이프라인 장애에 견디는 구조로 승격됐다.
