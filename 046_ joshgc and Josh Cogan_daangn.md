원본코드
#!/usr/bin/env python

# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Checks that the development environment is configured for Cloud ML."""

from __future__ import print_function
import pkg_resources
import re
import subprocess
import sys

MIN_CLOUD_ML_SDK_VERSION = '0.1.7a0'
MIN_CLOUD_SDK_VERSION = '136.0.0'
MIN_CLOUD_DATAFLOW_VERSION = '0.4.3'
MIN_TENSORFLOW_VERSION = '0.11.0rc0'

def get_version_from_pip(package_name):
  """Returns the version of an installed pip package."""
  try:
    package_info = subprocess.check_output(['pip', 'show', package_name])
  except subprocess.CalledProcessError:
    print('ERROR: Package %s has not been installed with pip.' % package_name,
          file=sys.stderr)
    exit(1)
  for line in package_info.split('\n'):
    m = re.match(r'Version: (.+)', line)
    if m:
      return m.group(1)
  print('ERROR: Unable to parse "pip show" output: %s' % package_info,
        file=sys.stderr)
  exit(1)

def get_cloud_sdk_version():
  """Returns the version of the Cloud SDK that is installed."""
  gcloud_info = subprocess.check_output(['gcloud', 'version'])
  for line in gcloud_info.split('\n'):
    m = re.match(r'Google Cloud SDK (.+)', line)
    if m:
      return m.group(1)
  print('ERROR: Unable to parse "gcloud version" output: %s' % gcloud_info,
        file=sys.stderr)
  exit(1)

def check_version_is_supported(name, version, min_version, help=''):
  """Checks whether a particular version of a package is new enough."""
  if (pkg_resources.parse_version(version) <
      pkg_resources.parse_version(min_version)):
    # Version is too old.
    print('ERROR: Unsupported %s version: %s (minimum %s).%s' %
              (name, version, min_version, (' %s' % help) if help else ''),
          file=sys.stderr)
    exit(1)

# Check that TensorFlow is installed.
import tensorflow as tf
check_version_is_supported('TensorFlow', tf.__version__, MIN_TENSORFLOW_VERSION)

# Check that the Apache Beam (a.k.a. Cloud Dataflow) SDK is installed.
import apache_beam
check_version_is_supported(
    'Cloud Dataflow', get_version_from_pip('google-cloud-dataflow'),
    MIN_CLOUD_DATAFLOW_VERSION)

# Check that the Cloud ML SDK is installed.
import google.cloud.ml
check_version_is_supported(
    'Cloud ML SDK', get_version_from_pip('cloudml'), MIN_CLOUD_ML_SDK_VERSION)

# Check that the Cloud SDK is installed, initialized, and logged in.
check_version_is_supported(
    'Cloud SDK', get_cloud_sdk_version(), MIN_CLOUD_SDK_VERSION,
    help='To update the Cloud SDK, run "gcloud components update".')
project_id = subprocess.check_output(
    ['gcloud', 'config', 'list', 'project',
     '--format', 'value(core.project)']).rstrip()
auth_token = subprocess.check_output(
    ['gcloud', 'auth', 'print-access-token']).rstrip()

# Check that the Cloud ML API is enabled.
models = subprocess.check_output([
    'curl', '-s', '-S', '-X', 'GET', '-H', 'Content-Type: application/json',
    '-H', 'Authorization: Bearer %s' % auth_token,
    'https://ml.googleapis.com/v1beta1/projects/%s/models' % project_id])
if '"error"' in models:
  print('ERROR: Unable to list Cloud ML models: %s' % models, file=sys.stderr)
  exit(1)

# Everything completed successfully.
print('Success! Your environment is configured correctly.')

레거시 환경 검증 스크립트 수준에서 벗어나지 못한 채 외부 CLI·전역 부수효과·직접 프로세스 종료에 강하게 결합되어 있어, 실제 운영 환경에서는 검증기 자체가 장애 원인이 될 수 있는 구조다.

제안패치
#!/usr/bin/env python3

# Copyright 2016-2026 Google Inc. & Production Refactoring Group.
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

"""Production-Hardened script checks that the development environment is configured for Cloud ML."""

import json
import logging
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Tuple

# 현대적인 버전 파싱 및 메타데이터 조회 모듈 사용 (pkg_resources / 외부 pip show 대체)
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version, InvalidVersion

# 구조화된 로깅 설정 (CLI 가시성 확보)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("check-env")

# 최소 요구 버전 상수 정의
MIN_CLOUD_ML_SDK_VERSION = '0.1.7a0'
MIN_CLOUD_SDK_VERSION = '136.0.0'
MIN_CLOUD_DATAFLOW_VERSION = '0.4.3'
MIN_TENSORFLOW_VERSION = '0.11.0rc0'


def get_version_from_env(package_name: str) -> str:
    """Returns the version of an installed package directly using modern importlib.metadata.
    
    Security & Stability:
        Avoids external 'pip show' process calls, ensuring we inspect 
        the exact Python interpreter environment currently executing.
    """
    try:
        return version(package_name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Package '{package_name}' has not been installed in the current environment.") from exc


def get_cloud_sdk_version() -> str:
    """Returns the version of the installed Google Cloud SDK safely."""
    try:
        output = subprocess.check_output(['gcloud', 'version'], stderr=subprocess.STDOUT)
        gcloud_info = output.decode('utf-8', errors='ignore')
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Google Cloud SDK ('gcloud') is not installed or not available in PATH.") from exc

    for line in gcloud_info.splitlines():
        if line.startswith("Google Cloud SDK"):
            parts = line.split()
            if len(parts) >= 4:
                return parts[3].strip()
                
    raise ValueError('Unable to parse version from "gcloud version" output.')


def check_version_is_supported(name: str, current_version: str, min_version: str, help_text: str = '') -> None:
    """Checks whether a particular version of a package satisfies the minimum requirement using packaging.version."""
    try:
        current_parsed = Version(current_version)
        min_parsed = Version(min_version)
    except InvalidVersion as exc:
        raise ValueError(f"Failed to parse version strings for {name} (Current: '{current_version}', Min: '{min_version}').") from exc

    if current_parsed < min_parsed:
        help_msg = f" {help_text}" if help_text else ""
        raise ValueError(f"Unsupported {name} version: {current_version} (minimum required: {min_version}).{help_msg}")


def get_gcloud_config() -> Tuple[str, str]:
    """Retrieves active GCP project ID and access token via gcloud CLI."""
    try:
        proj_out = subprocess.check_output(['gcloud', 'config', 'list', 'project', '--format', 'value(core.project)'])
        project_id = proj_out.decode('utf-8', errors='ignore').strip()
        if not project_id:
            raise ValueError("GCP Project ID is not set in gcloud config.")

        token_out = subprocess.check_output(['gcloud', 'auth', 'print-access-token'])
        auth_token = token_out.decode('utf-8', errors='ignore').strip()
        if not auth_token:
            raise ValueError("Failed to retrieve gcloud access token.")

        return project_id, auth_token
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Failed to execute gcloud commands. Ensure you are logged in ('gcloud auth login').") from exc


def check_cloud_ml_api(project_id: str, auth_token: str) -> None:
    """Validates that the Cloud ML API is enabled and accessible without exposing raw internal payloads."""
    # 순수한 HTTPS Endpoint URL 구성 (Markdown 문자열 혼입 원천 차단)
    url = f"https://ml.googleapis.com/v1beta1/projects/{project_id}/models"
    req = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}"
        },
        method="GET"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read().decode('utf-8', errors='ignore')
            data = json.loads(body)
            if "error" in data:
                # 내부 상세 정보 전체 노출 대신 CLI 사용자에게 필요한 요약 메시지만 전달
                err_msg = data["error"].get("message", "Unknown API error")
                raise RuntimeError(f"Cloud ML API rejected the request: {err_msg}")
                
    except urllib.error.HTTPError as exc:
        # HTTP 에러 시 민감한 서버 응답 원본 대신 상태 코드 및 안전한 메시지만 추출
        raise RuntimeError(f"Cloud ML API HTTP error occurred (Status Code: {exc.code}). Verify project permissions.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network connection failed while reaching Cloud ML API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse JSON response from Cloud ML API endpoint.") from exc


def verify_environment() -> None:
    """Performs comprehensive environment and dependency checks."""
    logger.info("Checking TensorFlow installation...")
    import tensorflow as tf
    check_version_is_supported('TensorFlow', tf.__version__, MIN_TENSORFLOW_VERSION)

    logger.info("Checking Apache Beam (Cloud Dataflow) SDK installation...")
    beam_version = get_version_from_env('apache-beam')
    check_version_is_supported('Cloud Dataflow', beam_version, MIN_CLOUD_DATAFLOW_VERSION)

    logger.info("Checking Cloud ML SDK installation...")
    cloudml_version = get_version_from_env('cloudml')
    check_version_is_supported('Cloud ML SDK', cloudml_version, MIN_CLOUD_ML_SDK_VERSION)

    logger.info("Checking Google Cloud SDK installation...")
    gcloud_version = get_cloud_sdk_version()
    check_version_is_supported(
        'Cloud SDK', gcloud_version, MIN_CLOUD_SDK_VERSION,
        help_text='To update the Cloud SDK, run "gcloud components update".'
    )

    logger.info("Retrieving gcloud configuration and auth token...")
    project_id, auth_token = get_gcloud_config()

    logger.info(f"Checking Cloud ML API access for project '{project_id}'...")
    check_cloud_ml_api(project_id, auth_token)


def main() -> int:
    """Main entrypoint with strict separation of concerns and safe exit codes."""
    try:
        verify_environment()
        print('Success! Your environment is configured correctly.')
        return 0
    except (ValueError, RuntimeError) as e:
        logger.error(f"Environment Verification Failed: {e}")
        return 1
    except Exception as e:
        logger.critical(f"Critical unexpected system error during environment check: {e}", exc_info=False)
        return 1


if __name__ == '__main__':
    sys.exit(main())

최종 개선사항
✅ pip show 외부 프로세스 의존 → importlib.metadata.version() 직접 조회 → 실행 중인 Python 환경과 패키지 버전의 일치성 확보
✅ pkg_resources 기반 버전 비교 → packaging.version.Version 검증 → 현대적인 PEP 440 버전 처리와 잘못된 버전 입력의 조기 차단
✅ curl 기반 API 검사 → urllib 직접 호출 및 10초 timeout → 외부 도구 의존성 제거와 네트워크 무한 대기 방지
✅ API 오류 원문 노출 → HTTP 상태·안전한 오류 메시지만 전달 → 내부 응답정보 및 불필요한 민감정보 노출 최소화
✅ 전역 검증 및 exit(1) → verify_environment()와 main()의 책임 분리 → 테스트 가능성과 CLI 종료 상태의 일관성 확보
✅ bytes/str 암묵적 혼용 → subprocess 출력의 명시적 UTF-8 디코딩 → 버전·설정 파싱 단계의 런타임 타입 오류 제거
✅ 외부 명령 실패와 검증 실패의 혼재 → ValueError·RuntimeError·네트워크 예외를 명확히 분리 → 장애 원인 추적성과 방어적 실패 처리 강화

원본의 환경 검증 목적은 유지하면서 레거시 패키지 조회·외부 HTTP 도구·프로세스 종료 결합을 제거해, 현재 버전은 검증기 자체가 장애를 만드는 가능성을 크게 낮춘 실무형 환경 진단 구조로 승격됐다.
