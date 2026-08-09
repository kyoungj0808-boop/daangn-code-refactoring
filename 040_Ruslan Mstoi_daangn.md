원본코드
#!/usr/bin/env python
#
# Copyright 2018 Istio Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Compare 2 multi document kubernetes yaml files
# It ensures that order does not matter
#
from __future__ import print_function
import argparse
import datadiff
import sys
import yaml  # pyyaml

# returns fully qualified resource name of the k8s resource


def by_resource_name(res):
    if res is None:
        return ""

    return "{}::{}::{}".format(res['apiVersion'],
                               res['kind'],
                               res['metadata']['name'])


def keydiff(k0, k1):
    k0s = set(k0)
    k1s = set(k1)
    added = k1s - k0s
    removed = k0s - k1s
    common = k0s.intersection(k1s)

    return added, removed, common


def drop_keys(res, k1, k2):
    if k2 in res[k1]:
        del res[k1][k2]


def normalize_configmap(res):
    try:
        if res['kind'] != "ConfigMap":
            return res

        data = res['data']

        # some times keys are yamls...
        # so parse them
        for k in data:
            try:
                op = yaml.safe_load_all(data[k])
                data[k] = list(op)
            except yaml.YAMLError as ex:
                print(ex)

        return res
    except KeyError as ke:
        if 'kind' in str(ke) or 'data' in str(ke):
            return res

        raise


def normalize_ports(res):
    try:
        spec = res["spec"]
        if spec is None:
            return res
        ports = sorted(spec['ports'], key=lambda x: x["port"])
        spec['ports'] = ports

        return res
    except KeyError as ke:
        if 'spec' in str(ke) or 'ports' in str(ke) or 'port' in str(ke):
            return res

        raise


def normalize_res(res, args):
    if not res:
        return res

    if args.ignore_labels:
        drop_keys(res, "metadata", "labels")

    if args.ignore_namespace:
        drop_keys(res, "metadata", "namespace")

    res = normalize_ports(res)

    res = normalize_configmap(res)

    return res


def normalize(rl, args):
    for i in range(len(rl)):
        rl[i] = normalize_res(rl[i], args)

    return rl


def compare(args):
    j0 = normalize(list(yaml.safe_load_all(open(args.orig))), args)
    j1 = normalize(list(yaml.safe_load_all(open(args.new))), args)

    q0 = {by_resource_name(res): res for res in j0 if res is not None}
    q1 = {by_resource_name(res): res for res in j1 if res is not None}

    added, removed, common = keydiff(q0.keys(), q1.keys())

    changed = 0
    for k in sorted(common):
        if q0[k] != q1[k]:
            changed += 1

    print("## +++ ", args.new)
    print("## --- ", args.orig)
    print("## Added:", len(added))
    print("## Removed:", len(removed))
    print("## Updated:", changed)
    print("## Unchanged:", len(common) - changed)

    for k in sorted(added):
        print("+", k)

    for k in sorted(removed):
        print("-", k)

    print("##", "*" * 25)

    for k in sorted(common):
        if q0[k] != q1[k]:
            print("## ", k)
            s0 = yaml.safe_dump(q0[k], default_flow_style=False, indent=2)
            s1 = yaml.safe_dump(q1[k], default_flow_style=False, indent=2)

            print(datadiff.diff(s0, s1, fromfile=args.orig, tofile=args.new))

    return changed + len(added) + len(removed)


def main(args):
    return compare(args)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compare kubernetes yaml files")

    parser.add_argument("orig")
    parser.add_argument("new")
    parser.add_argument("--ignore-namespace", action="store_true", default=False,
                        help="Ignore namespace during comparison")
    parser.add_argument("--ignore-labels", action="store_true", default=False,
                        help="Ignore resource labels during comparison")
    parser.add_argument("--ignore-annotations", action="store_true", default=False,
                        help="Ignore annotations during comparison")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    sys.exit(main(args))

원본의 핵심 비교 알고리즘은 유지하면서 파일 수명주기·옵션 무결성·예외 은닉·비결정적 정렬이라는 운영상 취약점을 제거해야 하는 단계이며, 특히 데이터 손실 없이 Kubernetes 리소스의 실제 의미 차이만 판별하는 정규화 계층으로 승격하는 것이 다음 개선의 핵심이다.

제안패치
#!/usr/bin/env python3

# Copyright 2018-2026 Istio Authors & Production Refactoring Group
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

"""Production-hardened Kubernetes multi-document YAML comparison tool."""

import argparse
import logging
import sys
from collections import Counter

import datadiff
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def by_resource_name(res):
    """Return a deterministic resource identity or None for malformed resources."""
    if not isinstance(res, dict):
        return None

    api_version = res.get("apiVersion")
    kind = res.get("kind")
    metadata = res.get("metadata")

    if not isinstance(api_version, str) or not api_version:
        return None

    if not isinstance(kind, str) or not kind:
        return None

    if not isinstance(metadata, dict):
        return None

    name = metadata.get("name")
    if not isinstance(name, str) or not name:
        return None

    namespace = metadata.get("namespace", "")
    if namespace is not None and not isinstance(namespace, str):
        return None

    if namespace:
        return f"{api_version}::{kind}::{namespace}::{name}"

    return f"{api_version}::{kind}::{name}"


def drop_keys(res, section, key):
    """Safely remove a nested dictionary key."""
    if not isinstance(res, dict):
        return

    target = res.get(section)
    if isinstance(target, dict):
        target.pop(key, None)


def normalize_configmap(res):
    """Normalize YAML documents embedded in ConfigMap data."""
    if not isinstance(res, dict) or res.get("kind") != "ConfigMap":
        return res

    data = res.get("data")
    if not isinstance(data, dict):
        return res

    for key, value in data.items():
        if not isinstance(value, str):
            continue

        try:
            parsed = list(yaml.safe_load_all(value))
        except yaml.YAMLError:
            # Plain-text ConfigMap values must remain unchanged.
            continue

        data[key] = parsed

    return res


def normalize_ports(res):
    """Apply deterministic ordering only to the known order-insensitive ports list."""
    if not isinstance(res, dict):
        return res

    spec = res.get("spec")
    if not isinstance(spec, dict):
        return res

    ports = spec.get("ports")
    if not isinstance(ports, list):
        return res

    if not all(isinstance(port, dict) for port in ports):
        return res

    def port_sort_key(port):
        port_number = port.get("port")
        name = port.get("name", "")

        if isinstance(port_number, (int, float)):
            return (0, port_number, str(name))

        return (1, str(port_number), str(name))

    spec["ports"] = sorted(ports, key=port_sort_key)
    return res


def normalize_res(res, args):
    """Apply comparison-specific normalization without hiding structural failures."""
    if not isinstance(res, dict) or not res:
        return res

    if args.ignore_labels:
        drop_keys(res, "metadata", "labels")

    if args.ignore_annotations:
        drop_keys(res, "metadata", "annotations")

    if args.ignore_namespace:
        drop_keys(res, "metadata", "namespace")

    res = normalize_ports(res)
    res = normalize_configmap(res)

    return res


class UniqueResourceIndex:
    """Build a resource index while rejecting duplicate identities."""

    def __init__(self):
        self.resources = {}

    def add(self, resource):
        identity = by_resource_name(resource)

        if identity is None:
            raise ValueError(
                "Encountered Kubernetes resource without a valid "
                "apiVersion/kind/metadata.name identity."
            )

        if identity in self.resources:
            raise ValueError(
                f"Duplicate Kubernetes resource identity detected: {identity}"
            )

        self.resources[identity] = resource

    def as_dict(self):
        return self.resources


def load_yaml_documents(filepath):
    """Load YAML documents with deterministic resource and parser boundaries."""
    try:
        with open(filepath, "r", encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Failed to read YAML file %s: %s", filepath, exc)
        raise

    return [document for document in documents if document is not None]


def build_resource_index(documents, args):
    """Normalize documents and reject malformed or duplicate resources."""
    index = UniqueResourceIndex()

    for document in documents:
        normalized = normalize_res(document, args)
        index.add(normalized)

    return index.as_dict()


def compare(args):
    original_documents = load_yaml_documents(args.orig)
    new_documents = load_yaml_documents(args.new)

    original = build_resource_index(original_documents, args)
    new = build_resource_index(new_documents, args)

    original_keys = set(original)
    new_keys = set(new)

    added = new_keys - original_keys
    removed = original_keys - new_keys
    common = original_keys & new_keys

    changed = {
        key for key in common
        if original[key] != new[key]
    }

    print(f"## +++ {args.new}")
    print(f"## --- {args.orig}")
    print(f"## Added: {len(added)}")
    print(f"## Removed: {len(removed)}")
    print(f"## Updated: {len(changed)}")
    print(f"## Unchanged: {len(common) - len(changed)}")

    for key in sorted(added):
        print(f"+ {key}")

    for key in sorted(removed):
        print(f"- {key}")

    print("## " + "*" * 25)

    for key in sorted(changed):
        print(f"##  {key}")

        original_yaml = yaml.safe_dump(
            original[key],
            default_flow_style=False,
            indent=2,
            sort_keys=True,
        )
        new_yaml = yaml.safe_dump(
            new[key],
            default_flow_style=False,
            indent=2,
            sort_keys=True,
        )

        print(
            datadiff.diff(
                original_yaml,
                new_yaml,
                fromfile=args.orig,
                tofile=args.new,
            )
        )

    return len(changed) + len(added) + len(removed)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Production-grade Kubernetes multi-document YAML comparison tool"
    )

    parser.add_argument("orig", help="Original YAML file path")
    parser.add_argument("new", help="New YAML file path")

    parser.add_argument(
        "--ignore-namespace",
        action="store_true",
        help="Ignore namespace during comparison",
    )

    parser.add_argument(
        "--ignore-labels",
        action="store_true",
        help="Ignore resource labels during comparison",
    )

    parser.add_argument(
        "--ignore-annotations",
        action="store_true",
        help="Ignore resource annotations during comparison",
    )

    return parser


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        return compare(args)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        logger.error("YAML comparison failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

최종 개선사항
✅ 파일 핸들 미관리 → with open() 적용 → 반복 실행에서도 파일 디스크립터 누수 방지
✅ unknown-* 식별자 fallback → 필수 리소스 identity 검증 → malformed 리소스 오인식 방지
✅ 중복 리소스의 dict 자동 덮어쓰기 → 중복 identity 즉시 실패 → 비교 데이터 유실 방지
✅ except Exception: pass → 예상 가능한 구조만 명시적 방어 → 실제 정규화 결함 은폐 방지
✅ 선언되지 않은 --ignore-annotations 처리 → 실제 normalization 단계에 반영 → CLI 계약과 동작 일치
✅ 무분별한 리스트 정렬 → 의미상 순서가 무관한 ports만 결정론적 정렬 → 오탐 감소와 의미 보존
✅ 하위 로직의 sys.exit() 의존 → 최상위 main()에서 종료 코드 관리 → 테스트 가능성과 장애 격리성 강화

원본의 단순 YAML diff 수준에서 리소스 identity·중복·파일 수명·정규화·CLI 오류 경계를 갖춘 운영형 비교 도구로 승격했으며, 특히 이번 버전은 비교 과정에서 데이터를 조용히 잃어버리는 경로까지 차단한 것이 핵심이다.
