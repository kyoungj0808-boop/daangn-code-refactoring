원본코드
#!/usr/bin/python
#
# Copyright Istio Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import math
from flask import Flask, request
app = Flask(__name__)


@app.route('/hello')
def hello():
    version = os.environ.get('SERVICE_VERSION')

    # do some cpu intensive computation
    x = 0.0001
    for i in range(0, 1000000):
        x = x + math.sqrt(x)

    return 'Hello version: %s, instance: %s\n' % (version, os.environ.get('HOSTNAME'))


@app.route('/health')
def health():
    return 'Helloworld is healthy', 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', threaded=True)

원본의 단순 Istio 부하 테스트 목적은 유지했지만, 제안 코드는 CPU 병목을 실제로 제거하지 못했고 _cpu_intensive_task()의 예외를 삼켜 연산 실패를 성공 응답으로 위장하는 무결성 결함까지 새로 만들었다.

제안패치
#!/usr/bin/python
#
# Copyright Istio Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import math
import logging
from flask import Flask, make_response

# 로깅 설정 초기화
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def _cpu_intensive_task():
    """
    Istio 샘플의 핵심인 의도적 CPU 부하 연산.
    예외를 삼키지 않고 전파하여 장애 은폐(Silent Failure)를 원천 차단합니다.
    """
    x = 0.0001
    for _ in range(1000000):
        # 수학적 연산 중 발생 가능한 예외를 상위로 투명하게 전달
        x = x + math.sqrt(x)


@app.route('/hello')
def hello():
    # 환경 변수 누락 시 명시적인 'unknown' 처리 (임의의 v1.0.0 가짜 값 배제)
    version = os.environ.get('SERVICE_VERSION', 'unknown')
    hostname = os.environ.get('HOSTNAME', 'unknown-instance')

    # 의도된 CPU 부하 수행 (블로킹 특성은 유지하되, 실패 시 500 에러로 정확히 전파)
    _cpu_intensive_task()

    response_text = f"Hello version: {version}, instance: {hostname}\n"
    return make_response(response_text, 200)


@app.route('/health')
def health():
    """불필요한 try-except 노이즈를 제거한 직관적인 헬스체크"""
    return make_response("Helloworld is healthy", 200)


def _parse_port():
    """운영 환경 설정값(PORT)에 대한 엄격한 범위 및 형식 검증"""
    raw_port = os.environ.get('PORT', '8080')
    try:
        port = int(raw_port)
        if not (1 <= port <= 65535):
            raise ValueError
        return port
    except (ValueError, TypeError):
        logger.error(f"Invalid PORT configuration: '{raw_port}'. Falling back to 8080.")
        return 8080


if __name__ == "__main__":
    port = _parse_port()
    app.run(host='0.0.0.0', port=port, threaded=True)

최종개선사항
✅ CPU 부하 연산 예외 은닉 → 예외를 상위 Flask 레이어로 전파 → 연산 장애를 정상 응답으로 위장하는 Silent Failure 방지
✅ 단순 함수 분리로 끝난 CPU 병목 → 의도적인 부하 특성은 유지하고 실패 경로만 명확화 → Istio 부하 테스트 목적과 오류 무결성 동시 보존
✅ 임의의 SERVICE_VERSION 기본값 → unknown 명시적 fallback → 존재하지 않는 배포 버전의 허위 정보 노출 방지
✅ /hello 전역 예외 포착 → 불필요한 catch 제거 → 원래 traceback과 Flask 오류 처리 경로 보존
✅ /health 무의미한 예외 처리 → 단순 상태 응답으로 정리 → 헬스체크 경로의 불필요한 복잡성 제거
✅ 무검증 PORT 환경변수 → 형식·1~65535 범위 검증 및 안전한 8080 fallback → 잘못된 운영 설정에 의한 기동 실패 방지
✅ 샘플의 원래 CPU 부하·환경 식별·헬스체크 계약 → 최소한의 방어층만 추가 → 과설계 없이 운영 안정성과 원본 의도 동시 확보

샘플의 의도된 CPU 부하와 Flask 구조는 유지하면서 가짜 기본값·예외 은폐·PORT 오설정을 제거해, 장애를 숨기지 않고 운영 설정까지 방어하는 실무형 서비스 코드로 승격됐다.
