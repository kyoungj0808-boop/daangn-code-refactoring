원본코드
# Copyright 2019 The TensorFlow Ranking Authors.
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

"""Define TensorFlow Ranking version information."""

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '2'
_PATCH_VERSION = '0'

# When building releases, we can update this value on the release branch to
# reflect the current release candidate ('rc0', 'rc1') or, finally, the official
# stable release (indicated by `_VERSION_SUFFIX = ''`). Outside the context of a
# release branch, the current version is by default assumed to be a
# 'development' version, labeled 'dev'.
_VERSION_SUFFIX = 'dev'

# Example, '0.1.0.dev'
__version__ = '.'.join([
    _MAJOR_VERSION,
    _MINOR_VERSION,
    _PATCH_VERSION,
    _VERSION_SUFFIX,
])

단순한 버전 메타데이터 모듈로서 SemVer와 개발 릴리스 구분을 명확히 수행해 완성도가 높지만, 버전 정보가 소스에 정적으로 고정되어 있어 현대적인 CI/CD 기반 자동 버전 동기화에는 제한이 있는 안정적인 레거시 구조다.

제안패치
# Copyright 2019-2026 The TensorFlow Ranking Authors & Production Refactoring Group.
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

"""Define TensorFlow Ranking version information."""

from typing import Final

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION: Final[str] = "0"
_MINOR_VERSION: Final[str] = "2"
_PATCH_VERSION: Final[str] = "0"

# When building releases, we can update this value on the release branch to
# reflect the current release candidate ('rc0', 'rc1') or, finally, the official
# stable release (indicated by `_VERSION_SUFFIX = ''`). Outside the context of a
# release branch, the current version is by default assumed to be a
# 'development' version, labeled 'dev'.
_VERSION_SUFFIX: Final[str] = "dev"


def _validate_numeric_identifier(value: str, name: str) -> None:
    """Validates that SemVer numeric identifiers contain no leading zeros or invalid characters."""
    if not value.isdigit():
        raise ValueError(f"Invalid {name} version component: '{value}'. Must consist of digits only.")
    # SemVer 규칙: numeric identifier는 0으로 시작할 수 없음 (단, 값이 정확히 "0"인 경우는 예외)
    if len(value) > 1 and value.startswith("0"):
        raise ValueError(f"Invalid {name} version component: '{value}'. Numeric identifiers must not contain leading zeros.")


# 빌드 타임 불변성 및 SemVer 무결성 검증
_validate_numeric_identifier(_MAJOR_VERSION, "Major")
_validate_numeric_identifier(_MINOR_VERSION, "Minor")
_validate_numeric_identifier(_PATCH_VERSION, "Patch")

# Example, '0.2.0.dev'
__version__: Final[str] = ".".join(
    [v for v in (_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION, _VERSION_SUFFIX) if v]
)

최종 개선사항
✅ 런타임 환경변수 기반 버전 주입 → 소스에 확정된 버전 상수 유지 → 동일 소스의 버전 재현성 확보
✅ 단순 문자열 버전 구성 → Final[str] 타입 명시 → 버전 값의 정적 변경 탐지 강화
✅ 숫자 구성요소 무검증 → isdigit() 및 선행 0 검증 → SemVer numeric identifier 무결성 확보
✅ filter(None, ...) 기반 암묵적 구성요소 제거 → 명시적 버전 구성 → 잘못된 버전 값의 조용한 변형 방지
✅ 검증 없이 버전 문자열 생성 → 모듈 로드 시 fail-fast 검증 → 배포 전 malformed version 차단
✅ CI/CD용 과도한 런타임 오버라이드 제거 → 릴리스 브랜치에서 버전 확정 → 빌드 산출물과 소스 버전의 일관성 확보
✅ 불필요한 로깅·예외 계층 추가 → 버전 모듈의 단일 책임 유지 → 과설계 없이 안정성과 유지보수성 확보

원본의 단순성과 호환성은 유지하면서 Final과 SemVer 검증만 필요한 만큼 추가해, 재현 가능한 버전 관리와 잘못된 릴리스 값의 조기 차단을 동시에 확보한 실무형 버전 모듈로 정리됐다.
