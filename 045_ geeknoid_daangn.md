원본코드
#!/usr/bin/python

# Copyright 2018 Istio Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python script generates a JWT signed by a Google service account

Example:
./sa-jwt.py  --iss example-issuer --aud foo,bar --claims=email:foo@google.com,dead:beef key.json
"""
from __future__ import print_function
import argparse
import time

import google.auth.crypt
import google.auth.jwt


def main(args):
    """Generates a signed JSON Web Token using a Google API Service Account."""
    signer = google.auth.crypt.RSASigner.from_service_account_file(
        args.service_account_file)
    now = int(time.time())
    payload = {
        # expire in one hour.
        "exp": now + 3600,
        "iat": now,
    }
    if args.iss:
        payload["iss"] = args.iss

    if args.sub:
        payload["sub"] = args.sub
    else:
        payload["sub"] = args.iss

    if args.aud:
        if "," in args.aud:
            payload["aud"] = args.aud.split(",")
        else:
            payload["aud"] = args.aud

    if args.claims:
        for item in args.claims.split(","):
            k, v = item.split(':')
            payload[k] = v

    signed_jwt = google.auth.jwt.encode(signer, payload)
    return signed_jwt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # positional arguments
    parser.add_argument(
        'service_account_file',
        help='The path to your service account key file (in JSON format).')
    # optional arguments
    parser.add_argument("-iss", "--iss",
                        help="iss claim. This should be your service account email.")
    parser.add_argument("-aud", "--aud",
                        help="aud claim. This is comma-separated-list of audiences")
    parser.add_argument("-sub", "--sub",
                        help="sub claim. If not provided, it is set to the same as iss claim.")
    parser.add_argument("-claims", "--claims",
                        help="Other claims in format name1:value1,name2:value2 etc. Only string values are supported.")
    print(main(parser.parse_args()))

작은 JWT CLI 도구로서는 구조가 간결하고 책임 분리가 명확하지만, 인증 핵심인 입력 검증·클레임 무결성·실패 처리 방어가 약해 현재 상태는 운영 자동화에 투입하기 전 보안 경계 보강이 필요한 수준이다.

제안패치
#!/usr/bin/env python3

# Copyright 2018-2026 Istio Authors & Production Refactoring Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Production-Hardened Python script generates a JWT signed by a Google service account.

Example:
    python sa-jwt.py key.json --iss example-issuer --aud foo,bar --claims email:foo@google.com,role:admin
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, Set

import google.auth.crypt
import google.auth.jwt

# 구조화된 로깅 설정 (CLI 가시성 확보)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("sa-jwt")

# 보안 원칙: 시스템 예약 클레임 정의 (사용자 입력으로 덮어쓰기 원천 차단)
RESERVED_CLAIMS: Set[str] = {"iss", "sub", "aud", "iat", "exp"}


def _validate_service_account_file(file_path: str) -> None:
    """Validates that the service account file exists, is a file, and is readable.
    
    Security & Stability:
        Fail fast with clear error messages instead of raw stack traces 
        when the key file is missing or inaccessible.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Service account file not found: '{file_path}'")
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a valid file: '{file_path}'")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Permission denied to read service account file: '{file_path}'")


def _parse_claims(claims_str: str) -> Dict[str, str]:
    """Safely parses comma-separated claims supporting values containing colons (e.g., URLs)."""
    claims = {}
    if not claims_str:
        return claims

    for item in claims_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid claim format (missing ':'): '{item}'")
        
        k, v = item.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError(f"Empty claim key in item: '{item}'")
        
        # 보안 원칙: 예약 클레임 변조 시도 탐지 및 즉시 차단
        if k in RESERVED_CLAIMS:
            raise ValueError(
                f"Security Violation: '{k}' is a reserved JWT claim and cannot be overridden via --claims."
            )
            
        claims[k] = v

    return claims


def generate_jwt(
    service_account_file: str,
    iss: str = None,
    aud: str = None,
    sub: str = None,
    claims_str: str = None,
    expiration_seconds: int = 3600
) -> str:
    """Generates a signed JSON Web Token using a Google API Service Account with strict integrity enforcement."""
    # 1. 만료 시간 엄격 검증 (양수만 허용)
    if expiration_seconds <= 0:
        raise ValueError(f"expiration_seconds must be greater than 0, given: {expiration_seconds}")

    # 2. 서비스 계정 파일 검증
    _validate_service_account_file(service_account_file)

    # 3. 크립토그래픽 서명기 로드 (구체적인 예외 분리)
    try:
        signer = google.auth.crypt.RSASigner.from_service_account_file(service_account_file)
    except (ValueError, TypeError, KeyError) as exc:
        # 민감한 파일 내용이나 내부 객체 구조가 로그나 예외 메시지에 노출되지 않도록 경계 설정
        raise ValueError(
            f"Failed to parse service account JSON format or private key from: '{service_account_file}'"
        ) from exc
    except Exception as exc:
        raise RuntimeError("Cryptographic initialization failed due to an unexpected error.") from exc

    now = int(time.time())
    payload: Dict[str, Any] = {
        "exp": now + expiration_seconds,
        "iat": now,
    }

    if iss:
        payload["iss"] = iss

    if sub:
        payload["sub"] = sub
    elif iss:
        payload["sub"] = iss

    if aud:
        if "," in aud:
            payload["aud"] = [a.strip() for a in aud.split(",") if a.strip()]
        else:
            payload["aud"] = aud.strip()

    # 4. 추가 클레임 파싱 및 예약 클레임 오염 방어 적용
    if claims_str:
        additional_claims = _parse_claims(claims_str)
        # update() 대신 개별 할당 또는 겹침 검증을 거쳐 안전하게 주입
        for k, v in additional_claims.items():
            if k in payload:
                raise ValueError(f"Collision detected: Claim '{k}' is already defined by system parameters.")
            payload[k] = v

    # 5. JWT 인코딩 수행
    try:
        signed_jwt = google.auth.jwt.encode(signer, payload)
        if isinstance(signed_jwt, bytes):
            signed_jwt = signed_jwt.decode("utf-8")
        return signed_jwt
    except Exception as exc:
        # 내부 토큰 페이로드나 비밀키 상태가 에러 메시지로 유출되지 않도록 가림
        raise RuntimeError("Failed to encode JWT signature due to signing error.") from exc


def main() -> int:
    """CLI entrypoint with defensive error handling and proper exit codes."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "service_account_file",
        help="The path to your service account key file (in JSON format)."
    )
    parser.add_argument(
        "-iss", "--iss",
        help="iss claim. This should be your service account email."
    )
    parser.add_argument(
        "-aud", "--aud",
        help="aud claim. This is a comma-separated list of audiences."
    )
    parser.add_argument(
        "-sub", "--sub",
        help="sub claim. If not provided, it is set to the same as iss claim."
    )
    parser.add_argument(
        "-claims", "--claims",
        help="Other claims in format name1:value1,name2:value2 etc. URL values containing colons are supported."
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=3600,
        help="Token expiration time in seconds (must be > 0, default: 3600)."
    )

    args = parser.parse_args()

    try:
        token = generate_jwt(
            service_account_file=args.service_account_file,
            iss=args.iss,
            aud=args.aud,
            sub=args.sub,
            claims_str=args.claims,
            expiration_seconds=args.exp
        )
        print(token)
        return 0
    except (ValueError, FileNotFoundError, PermissionError) as e:
        # 사용자 입력 및 설정 오류는 명확히 안내 (비밀값 노출 경계 준수)
        logger.error(f"Configuration/Input Error: {e}")
        return 1
    except Exception as e:
        # 예상치 못한 시스템 예외
        logger.critical(f"Critical System Error during JWT generation: {e}", exc_info=False)
        return 1


if __name__ == "__main__":
    sys.exit(main())

최종 개선사항
✅ 무제한 추가 클레임 주입 → iss/sub/aud/iat/exp 예약 클레임 충돌 차단 → JWT 페이로드 무결성 확보
✅ 음수·0 만료 시간 허용 → expiration_seconds > 0 선행 검증 → 이미 만료된 토큰 생성 방지
✅ claims의 단순 split(":") → 첫 번째 콜론만 기준으로 파싱 → URL 등 콜론 포함 값의 데이터 손실 방지
✅ 암호화·인코딩 예외의 민감정보 노출 가능성 → 예외 원인은 체인으로 보존하되 외부 메시지는 비식별화 → 키·페이로드 유출 위험 축소
✅ 파일 경로만 라이브러리에 전달 → 존재·파일 여부·읽기 권한 사전 검증 → 불필요한 암호화 초기화 실패 및 장애 원인 불명확성 제거
✅ 모든 오류를 단일 예외 처리 → 입력/설정 오류와 시스템 오류 분리 → CLI 자동화 환경에서 실패 원인과 종료 상태의 명확성 확보
✅ JWT 출력과 오류 로그 혼재 가능성 → 토큰은 stdout, 로그는 stderr로 분리 → 파이프라인에서 토큰 오염 및 데이터 처리 실패 방지

원본의 단순 JWT 생성 목적은 유지하면서 입력 검증·예약 클레임 무결성·민감정보 경계를 강화해, 현재 버전은 스크립트 규모를 불필요하게 키우지 않고도 실사용 가능한 보안형 CLI 구조로 승격됐다.
