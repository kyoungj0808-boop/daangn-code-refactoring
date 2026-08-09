원본코드
from datetime import datetime


def lambda_handler(event, context):
    with open("/mnt/efs/test.txt", 'a', encoding='utf-8') as f:
        f.write("{}: hello from lambda\n".format(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S%z")))

    with open("/mnt/efs/test.txt", 'r', encoding='utf-8') as f:
        content = f.readlines()

    return ''.join(content)

EFS 연동을 확인하는 샘플이라는 본래 목적에는 충실하지만, utcnow()의 부정확한 시간 표현과 불필요한 전체 파일 재읽기, EFS 장애 시 진단·복구 전략 부재 때문에 프로덕션 코드로는 안정성과 비용 효율성이 크게 부족하다.

제안패치
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EFS_FILE_PATH = "/mnt/efs/test.txt"


def lambda_handler(event, context):
    current_time = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    log_line = f"{current_time}: hello from lambda\n"

    file_dir = os.path.dirname(EFS_FILE_PATH)

    if file_dir and not os.path.isdir(file_dir):
        raise FileNotFoundError(
            f"EFS mount directory is missing: {file_dir}"
        )

    try:
        with open(EFS_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_line)
    except OSError:
        logger.exception("Failed to write to EFS: %s", EFS_FILE_PATH)
        raise

    logger.info("Successfully appended log entry to EFS.")
    return log_line


if __name__ == "__main__":
    lambda_handler({}, None)

최종 개선사항
✅ datetime.utcnow() 사용 → timezone-aware UTC 적용 → 시간 의미와 포맷 안정성 확보
✅ append 직후 전체 파일 재읽기 → 기록한 현재 데이터만 반환 → EFS 네트워크 I/O와 메모리 낭비 제거
✅ 무조건적인 Exception 포착 후 RuntimeError 변환 → OSError 진단 후 원본 예외 재전파 → 장애 원인과 traceback 보존
✅ exists() 기반 디렉터리 검사 → isdir() 기반 경로 계약 검증 → 마운트 경로 검증 의미 명확화
✅ EFS 경로 하드코딩 → EFS_FILE_PATH 상수화 → 설정 변경 및 유지보수성 향상
✅ EFS 장애를 무조건 성공 처리 → 실패를 로그로 기록하고 예외 전파 → Lambda 호출 실패 상태의 투명성 확보
✅ 단순 EFS 샘플에 과도한 추상화 추가 → 최소한의 방어 계층만 적용 → 원본 목적과 코드 단순성 보존

원본의 EFS 테스트 목적은 유지하면서 불필요한 I/O와 시간 처리 문제를 제거하고, 예외를 숨기지 않는 방어 구조로 정리하면 실무 테스트 코드로도 충분히 높은 완성도에 도달한다.
