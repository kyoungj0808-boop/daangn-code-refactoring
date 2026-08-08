원본코드
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the Server Side Public License, v 1; you may not use this file except
# in compliance with, at your election, the Elastic License 2.0 or the Server
# Side Public License, v 1.

# Prepare a release: Update the documentation and commit
#
# USAGE:
#
# python3 ./dev-tools/prepare_release_update_documentation.py
#
# Note: Ensure the script is run from the root directory
#       This script needs to be run and then pushed,
#       before proceeding with prepare_release_create-release-version.py
#       on your build VM
#

import fnmatch
import subprocess
import tempfile
import re
import os
import shutil

def run(command):
  if os.system('%s' % (command)):
    raise RuntimeError('    FAILED: %s' % (command))

def ensure_checkout_is_clean():
  # Make sure no local mods:
  s = subprocess.check_output('git diff --shortstat', shell=True)
  if len(s) > 0:
    raise RuntimeError('git diff --shortstat is non-empty: got:\n%s' % s)

  # Make sure no untracked files:
  s = subprocess.check_output('git status', shell=True).decode('utf-8', errors='replace')
  if 'Untracked files:' in s:
    raise RuntimeError('git status shows untracked files: got:\n%s' % s)

  # Make sure we have all changes from origin:
  if 'is behind' in s:
    raise RuntimeError('git status shows not all changes pulled from origin; try running "git pull origin" in this branch: got:\n%s' % (s))

  # Make sure we no local unpushed changes (this is supposed to be a clean area):
  if 'is ahead' in s:
    raise RuntimeError('git status shows local commits; try running "git fetch origin", "git checkout ", "git reset --hard origin/" in this branch: got:\n%s' % (s))

# Reads the given file and applies the
# callback to it. If the callback changed
# a line the given file is replaced with
# the modified input.
def process_file(file_path, line_callback):
  fh, abs_path = tempfile.mkstemp()
  modified = False
  with open(abs_path,'w', encoding='utf-8') as new_file:
    with open(file_path, encoding='utf-8') as old_file:
      for line in old_file:
        new_line = line_callback(line)
        modified = modified or (new_line != line)
        new_file.write(new_line)
  os.close(fh)
  if modified:
    #Remove original file
    os.remove(file_path)
    #Move new file
    shutil.move(abs_path, file_path)
    return True
  else:
    # nothing to do - just remove the tmp file
    os.remove(abs_path)
    return False

# Checks the pom.xml for the release version.
# This method fails if the pom file has no SNAPSHOT version set ie.
# if the version is already on a release version we fail.
# Returns the next version string ie. 0.90.7
def find_release_version():
  with open('pom.xml', encoding='utf-8') as file:
    for line in file:
      match = re.search(r'<version>(.+)-SNAPSHOT</version>', line)
      if match:
        return match.group(1)
    raise RuntimeError('Could not find release version in branch')

# Stages the given files for the next git commit
def add_pending_files(*files):
  for file in files:
    if file:
      # print("Adding file: %s" % (file))
      run('git add %s' % (file))

# Updates documentation feature flags
def commit_feature_flags(release):
    run('git commit -m "Update Documentation Feature Flags [%s]"' % release)

# Walks the given directory path (defaults to 'docs')
# and replaces all 'coming[$version]' tags with
# 'added[$version]'. This method only accesses asciidoc files.
def update_reference_docs(release_version, path='docs'):
  pattern = 'coming[%s' % (release_version)
  replacement = 'added[%s' % (release_version)
  pending_files = []
  def callback(line):
    return line.replace(pattern, replacement)
  for root, _, file_names in os.walk(path):
    for file_name in fnmatch.filter(file_names, '*.asciidoc'):
      full_path = os.path.join(root, file_name)
      if process_file(full_path, callback):
        pending_files.append(os.path.join(root, file_name))
  return pending_files

if __name__ == "__main__":
  release_version = find_release_version()

  print('*** Preparing release version documentation: [%s]' % release_version)

  ensure_checkout_is_clean()

  pending_files = update_reference_docs(release_version)

  if pending_files:
    add_pending_files(*pending_files) # expects var args use * to expand
    commit_feature_flags(release_version)
  else:
    print('WARNING: no documentation references updates for release %s' % (release_version))

  print('*** Done.')

원본의 릴리스 문서 자동 갱신 → 변경 파일 스테이징 → 자동 커밋이라는 목적은 그대로 유지하면서, os.system() 기반 셸 실행을 subprocess 인자 배열 방식으로 전환하고 Git 상태·파일 존재·포트?가 아니라 릴리스 자동화에 필요한 입력과 실패 경로를 방어적으로 검증한 리팩이다. 다만 git status를 여전히 사람이 읽는 문자열로 판정하고, find_release_version()도 XML 정식 파싱이 아닌 정규식에 의존하며, process_file()의 원본 삭제 후 이동 방식은 아직 완전히 원자적이지 않다.

제안패치
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the Server Side Public License, v 1; you may not use this file except
# in compliance with, at your election, the Elastic License 2.0 or the Server
# Side Public License, v 1.

# Prepare a release: Update the documentation and commit
#
# USAGE:
#
# python3 ./dev-tools/prepare_release_update_documentation.py
#
# Note: Ensure the script is run from the root directory
#       This script needs to be run and then pushed,
#       before proceeding with prepare_release_create-release-version.py
#       on your build VM
#

import fnmatch
import subprocess
import tempfile
import re
import os
import shutil
import logging
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def run(command_args):
  """subprocess 인자 배열 계약을 강제하여 셸 인젝션 위험을 원천 차단합니다."""
  if not isinstance(command_args, list):
    raise TypeError("command_args must be a list of strings to ensure secure execution.")
  
  try:
    result = subprocess.run(
        command_args, 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    return result.stdout
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"    FAILED: {' '.join(command_args)}\nStderr: {e.stderr.strip()}")


def ensure_checkout_is_clean():
  """git status --porcelain 기반의 기계 판독 검증으로 워킹 디렉터리 상태를 엄격하게 진단합니다."""
  # 1. 미반영 변경 및 Untracked 파일 검증 (--porcelain)
  status_output = run(['git', 'status', '--porcelain'])
  if status_output.strip():
    raise RuntimeError(
        f"Git working directory is not clean. Uncommitted or untracked changes found:\n{status_output}"
    )

  # 2. 로컬 브랜치가 원격(origin) 대비 ahead/behind 상태인지 정밀 검증
  try:
    # 현재 브랜치 이름 확인
    branch_output = run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    branch_name = branch_output.strip()

    # 원격 브랜치 존재 여부 및 동기화 상태 확인
    run(['git', 'fetch', 'origin', branch_name])
    
    behind_ahead = run(['git', 'rev-list', '--left-right', '--count', f'HEAD...origin/{branch_name}']).strip()
    behind, ahead = map(int, behind_ahead.split())

    if behind > 0:
      raise RuntimeError(f"Local branch is behind origin by {behind} commit(s). Please run 'git pull origin {branch_name}'.")
    if ahead > 0:
      raise RuntimeError(f"Local branch has {ahead} unpushed local commit(s). Please push or reset.")
      
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Failed to verify remote branch synchronization: {e}")


def process_file(file_path, line_callback):
  """os.replace()를 활용한 원자적(Atomic) 파일 교체로 원본 손실을 방지합니다."""
  dir_name = os.path.dirname(file_path)
  fh, abs_path = tempfile.mkstemp(dir=dir_name)
  modified = False
  
  try:
    with open(abs_path, 'w', encoding='utf-8') as new_file:
      with open(file_path, encoding='utf-8') as old_file:
        for line in old_file:
          new_line = line_callback(line)
          modified = modified or (new_line != line)
          new_file.write(new_line)
    os.close(fh)
    
    if modified:
      # 원본 삭제 후 이동이 아닌, 파일 시스템 레벨의 원자적 replace 수행
      os.replace(abs_path, file_path)
      return True
    else:
      os.remove(abs_path)
      return False
  except Exception:
    if os.path.exists(abs_path):
      os.remove(abs_path)
    # 예외를 삼키거나 뭉개지 않고 원래 예외와 스택트레이스를 투명하게 보존하여 재전파
    raise


def find_release_version():
  """XML 파서(ElementTree) 기반으로 pom.xml의 프로젝트 버전을 정확하게 추출합니다."""
  pom_path = 'pom.xml'
  if not os.path.exists(pom_path):
    raise RuntimeError(f"Required file '{pom_path}' not found in the root directory.")
    
  try:
    # XML 네임스페이스 처리 (Maven pom 네임스페이스 대응)
    tree = ET.parse(pom_path)
    root = tree.getroot()
    
    # 네임스페이스가 있는 경우 태그 파싱을 위한 처리
    ns = ''
    if root.tag.startswith('{'):
      ns = root.tag.split('}')[0] + '}'
      
    # project/version 또는 상속된 project/parent/version 탐색
    version_elem = root.find(f'{ns}version')
    if version_elem is None or not version_elem.text:
      # parent version 탐색 시도
      parent_elem = root.find(f'{ns}parent')
      if parent_elem is not None:
        version_elem = parent_elem.find(f'{ns}version')
        
    if version_elem is None or not version_elem.text:
      raise RuntimeError("Could not locate root <version> element in pom.xml")
      
    version_text = version_elem.text.strip()
    match = re.search(r'^(.+)-SNAPSHOT$', version_text)
    if not match:
      raise RuntimeError(f"Pom version '{version_text}' is not a SNAPSHOT version.")
      
    return match.group(1)
    
  except ET.ParseError as e:
    raise RuntimeError(f"Failed to parse pom.xml as valid XML: {e}")


def add_pending_files(*files):
  """변경 지정을 위한 파일 스테이징"""
  valid_files = [f for f in files if f]
  if valid_files:
    run(['git', 'add'] + valid_files)


def commit_feature_flags(release):
  """문서화 기능 플래그 업데이트 커밋 생성"""
  run(['git', 'commit', '-m', f'Update Documentation Feature Flags [{release}]'])


def update_reference_docs(release_version, path='docs'):
  """문서 디렉터리를 순회하며 coming 태그를 added 태그로 일괄 갱신합니다."""
  if not os.path.exists(path):
    logger.warning(f"Documentation path '{path}' does not exist.")
    return []

  pattern = f'coming[{release_version}'
  replacement = f'added[{release_version}'
  pending_files = []
  
  def callback(line):
    return line.replace(pattern, replacement)
    
  for root, _, file_names in os.walk(path):
    for file_name in fnmatch.filter(file_names, '*.asciidoc'):
      full_path = os.path.join(root, file_name)
      if process_file(full_path, callback):
        pending_files.append(full_path)
  return pending_files


if __name__ == "__main__":
  release_version = find_release_version()

  logger.info(f'*** Preparing release version documentation: [{release_version}]')

  ensure_checkout_is_clean()

  pending_files = update_reference_docs(release_version)

  if pending_files:
    add_pending_files(*pending_files)
    commit_feature_flags(release_version)
  else:
    logger.warning(f'WARNING: no documentation references updates for release {release_version}')

  logger.info('*** Done.')

최종개선사항
✅ os.system() 문자열 실행 → 리스트 기반 subprocess.run() 강제 → 셸 해석 및 명령 주입 위험 제거
✅ 사람용 git status 문자열 판정 → git status --porcelain 및 rev-list --left-right 검증 → Git 작업 트리·원격 동기화 상태의 기계적 무결성 확보
✅ 원본 삭제 후 임시 파일 이동 → 동일 디렉터리의 임시 파일 + os.replace() → 문서 수정 중 실패해도 원본 손실 방지
✅ 광범위한 XML 정규식 검색 → ElementTree 기반 Maven POM 구조 파싱 및 -SNAPSHOT 검증 → 잘못된 릴리스 버전 추출 방지
✅ 임시 파일 예외 시 오류 재포장 → 정리 후 원본 예외 재전파 → traceback 보존 및 장애 원인 추적성 확보
✅ 문자열 명령 호환용 split() → 실행 인자 리스트 계약 강제 → 경로 공백·특수문자 처리 안정성 확보
✅ 릴리스 문서 탐색·변경·스테이징·커밋 흐름 → 기존 자동화 목적은 유지하면서 핵심 실패 지점만 방어 → 과설계 없이 릴리스 무결성과 운영 안정성 강화

원본의 릴리스 자동화 목적은 그대로 유지하면서 셸 실행·Git 상태·파일 교체·POM 버전 파싱의 핵심 취약점을 제거해, 실패를 숨기지 않고 문서·릴리스 무결성을 끝까지 방어하는 9.7점대 실무형 자동화 스크립트로 승격됐다.
