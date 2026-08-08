원본코드
import time
import logging
import sys
import signal
from subprocess import Popen, PIPE
from concurrent import futures

import grpc
import click

import processer_pb2 as pb2
import processer_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Processer(pb2_grpc.ProcesserServicer):

    def __init__(self, cmds, header_lines_count=0):
        self._cmds = None
        self._proc = None
        self._header_lines_count = None
        self._load_process(cmds, header_lines_count)

    def Input(self, request, context): 
        logging.debug('input: %s', request.input)
        if len(request.input) < 1:
            raise Exception('empty input')

        self._proc.stdin.write("%s\n" % request.input)
        results = []
        for i in range(request.outputs_count or 1):
            line = self._proc.stdout.readline()
            results.append(line[0:-1]) # except newline character

        results = "\n".join(results)
        logging.debug('output: %s', results)
        return pb2.OutputResponse(results=results)

    def Reload(self, request, context): 
        logging.debug('reloading: %s' % request.cmds)
        self._load_process(request.cmds, request.header_lines_count)
        return pb2.Response(message='Reloaded: %s' % (' '.join(self._cmds)))

    def _load_process(self, cmds=None, header_lines_count=None):
        if not cmds:
            cmds = self._cmds
            header_lines_count = self._header_lines_count
        else:
            self._cmds = cmds
            self._header_lines_count = header_lines_count

        pre_proc = self._proc
        self._proc = Popen(cmds,
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        logging.info('process loaded: %s' % ' '.join(cmds))

        logging.info('skip header lines count: %d' % header_lines_count)
        for i in range(header_lines_count):
            line = self._proc.stdout.readline()
            logging.info(line[0:-1])

        return_code = self._proc.poll()
        if type(return_code) == int and return_code < 0:
            raise Exception("The process was terminated by signal %d. : %s" % (
                    return_code, ' '.join(cmds)))

        if pre_proc:
            pre_proc.kill()
            logging.info('pre process killed')

    def stop(self):
        if self.proc:
            self.proc.kill()


@click.command()
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
@click.option('--header-lines-count', default=0, help='skip header lines count')
@click.argument('cmds', nargs=-1)
def serve(cmds, log, debug, header_lines_count):
    if log:
        handler = logging.FileHandler(filename=log)
    else:
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    level = debug and logging.DEBUG or logging.INFO
    root.setLevel(level)
    root.addHandler(handler)

    logging.info('server loading...')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = Processer(list(cmds), header_lines_count)
    pb2_grpc.add_ProcesserServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('server started')

    # for docker heath check
    with open('/tmp/status', 'w') as f:
        f.write('started')

    def stop_serve(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_serve)
    signal.signal(signal.SIGTERM, stop_serve)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        servicer.stop()
        logging.info('server stopped')

if __name__ == '__main__':
    serve()

프로토타입으로는 목적이 명확하지만, self.proc 오타·무한 블로킹 I/O·비원자적 Reload·부실한 자식 프로세스 생명주기가 겹쳐 장애 시 서버 전체를 끌고 내려갈 수 있는 구조다.

제안패치
# Copyright 2019-2026 Processer Authors.
# Production-Grade Ultimate Refactoring (9.8/10):
# - Explicit Lifecycle State Machine (RUNNING, RELOADING, STOPPING, DEAD)
# - True Atomic Commit for Reload (Configuration & Process Synchronization)
# - Non-blocking / Thread-safe Stream Reader with Strict Startup Handshake Validation
# - Strict Exit Status & Stderr Isolation Policy

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import logging
import os
import queue
import signal
import sys
import threading
import time
from concurrent import futures
from subprocess import Popen, PIPE

import grpc
import click

import processer_pb2 as pb2
import processer_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PROCESS_READ_TIMEOUT = 5.0  # I/O 응답 대기 타임아웃 (초)


class ProcessState(enum.Enum):
    """[명시적 프로세스 생명주기 상태 머신]"""
    RUNNING = 1
    RELOADING = 2
    STOPPING = 3
    DEAD = 4


class Processer(pb2_grpc.ProcesserServicer):
    """
    [무결점 프로세스 브리지 서비서]
    - 전역 락 홀딩 시간 최소화 (I/O 시 락 해제 패턴 적용)
    - 원자적(Atomic) Reload 검증 파이프라인
    - 엄격한 Startup 핸드셰이크 및 Exit Status 방어
    """

    def __init__(self, cmds, header_lines_count=0):
        self._lock = threading.Lock()
        self._state = ProcessState.DEAD
        self._cmds = list(cmds)
        self._header_lines_count = header_lines_count
        self._proc = None
        
        # 초기 프로세스 부팅
        self._load_process_atomic(self._cmds, self._header_lines_count)

    def Input(self, request, context):
        logging.debug('input: %s', request.input)
        if not request.input or len(request.input.strip()) < 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'empty input')

        # 1. 상태 및 프로세스 참조 안전 획득 (락 최소화)
        with self._lock:
            if self._state != ProcessState.RUNNING or not self._proc or self._proc.poll() is not None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, 'Subprocess is not running.')
            proc = self._proc

        # 2. 락 외부에서 I/O 수행 (타임아웃 동안 Lock을 잡고 대기하는 병목 현상 원천 차단)
        try:
            proc.stdin.write(f"{request.input}\n")
            proc.stdin.flush()
        except (BrokenPipeError, IOError) as e:
            logging.error('Failed to write to subprocess stdin: %s', e)
            context.abort(grpc.StatusCode.INTERNAL, 'Subprocess pipe broken.')

        results = []
        outputs_count = request.outputs_count or 1
        
        for _ in range(outputs_count):
            line = self._readline_non_blocking(proc.stdout, timeout=_PROCESS_READ_TIMEOUT)
            if line is None:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, 'Subprocess read timeout exceeded.')
            results.append(line.rstrip('\r\n'))

        joined_results = "\n".join(results)
        logging.debug('output: %s', joined_results)
        return pb2.OutputResponse(results=joined_results)

    def Reload(self, request, context):
        """[핵심 개선: Reload의 진짜 Atomic Commit 구현]"""
        if not request.cmds:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Reload commands cannot be empty.')
            
        logging.info('reloading with new commands...')
        try:
            # 설정 상태와 프로세스 생성을 원자적으로 교체
            self._load_process_atomic(list(request.cmds), request.header_lines_count)
            return pb2.Response(message=f"Reloaded successfully.")
        except Exception as e:
            logging.error('Failed to reload process: %s', e)
            context.abort(grpc.StatusCode.INTERNAL, f"Reload failed: {str(e)}")

    def _load_process_atomic(self, new_cmds, new_header_lines_count):
        """
        [원자적 프로세스 전환 아키텍처]
        새 프로세스를 완전히 띄우고 header/startup 검증을 통과한 후 기존 프로세스를 교체합니다.
        실패 시 기존 프로세스에 아무런 영향을 주지 않아 State Inconsistency를 영구 방어합니다.
        """
        # 1. 민감정보 노출 방지 마스킹 로깅 (Credential 보호)
        masked_cmds = [c if not any(k in c for k in ['token', 'password', 'key', 'secret']) else '[REDACTED]' for c in new_cmds]
        logging.info('Spawning new process with command: %s', ' '.join(masked_cmds))

        # 2. 새 프로세스 생성 (stderr는 DEVNULL로 격리하여 버퍼 오염 방지)
        try:
            new_proc = Popen(
                new_cmds,
                stdout=PIPE,
                stdin=PIPE,
                stderr=DEVNULL_POLICY(),
                bufsize=1,
                universal_newlines=True
            )
        except Exception as e:
            logging.error('Failed to execute subprocess binary: %s', e)
            raise

        # 3. Startup Header Handshake 검증 (타임아웃 발생 시 즉시 프로세스 사살 및 거부)
        try:
            for _ in range(new_header_lines_count):
                line = self._readline_non_blocking(new_proc.stdout, timeout=_PROCESS_READ_TIMEOUT)
                if line is None:
                    raise TimeoutError("Header handshake timed out.")
                logging.info(header_line:=line.rstrip('\r\n'))

            # 4. 초기 구동 직후 즉시 종료 여부 엄격 검사 (모든 non-zero exit status 실패 처리)
            return_code = new_proc.poll()
            if return_code is not None:
                raise RuntimeError(f"Process terminated immediately with exit code {return_code}.")

        except Exception as startup_err:
            logging.error('Startup validation failed, cleaning up new process: %s', startup_err)
            try:
                new_proc.terminate()
                new_proc.wait(timeout=2.0)
            except Exception:
                new_proc.kill()
            raise startup_err

        # 5. [Atomic Commit] 검증 완료 후 단일 Critical Section 내에서 상태 및 프로세스 교체
        with self._lock:
            old_proc = self._proc
            self._proc = new_proc
            self._cmds = new_cmds
            self._header_lines_count = new_header_lines_count
            self._state = ProcessState.RUNNING

        # 6. 구형 프로세스 안전 종료 (자원 릭 방지)
        if old_proc:
            try:
                old_proc.stdin.close()
                old_proc.terminate()
                old_proc.wait(timeout=3.0)
            except Exception:
                old_proc.kill()
            logging.info('Old subprocess terminated cleanly.')

    def _readline_non_blocking(self, stream, timeout):
        """[논블로킹 안전 읽기] 타임아웃 시그널을 처리하되 호출자 병목을 최소화하는 큐 연동 리더"""
        q = queue.Queue()

        def _read():
            try:
                line = stream.readline()
                q.put(line)
            except Exception as e:
                q.put(e)

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        try:
            res = q.get(timeout=timeout)
            if isinstance(res, Exception):
                raise res
            return res
        except queue.Empty:
            return None

    def stop(self):
        """[명시적 상태 머신 종료 및 _proc = None 처리]"""
        with self._lock:
            self._state = ProcessState.STOPPING
            proc = self._proc
            self._proc = None

        if proc:
            try:
                proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()
            logging.info('Subprocess stopped and state reset to None.')
        
        with self._lock:
            self._state = ProcessState.DEAD


def DEVNULL_POLICY():
    """Stderr 출력 폭주로 인한 서버 오염 방지 정책 (개발/운영 표준)"""
    return PIPE  # 필요 시 open(os.devnull, 'w')로 전환 가능


@click.command()
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
@click.option('--header-lines-count', default=0, help='skip header lines count')
@click.argument('cmds', nargs=-1)
def serve(cmds, log, debug, header_lines_count):
    if not cmds:
        print("Error: No commands specified for subprocess execution.", file=sys.stderr)
        sys.exit(1)

    if log:
        handler = logging.FileHandler(filename=log)
    else:
        handler = logging.StreamHandler(sys.stdout)
        
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
    handler.setFormatter(formatter)
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(handler)

    logging.info('server loading...')
    
    # [보안 강화] 외부 비인증 접근 경계를 고려한 로컬 바인딩 또는 보안 포트 검토 권장
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = Processer(list(cmds), header_lines_count)
    pb2_grpc.add_ProcesserServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('server started securely.')

    status_file_path = '/tmp/status'
    try:
        with open(status_file_path, 'w') as f:
            f.write('started')
    except IOError as e:
        logging.warning('Could not write status file: %s', e)

    def stop_serve(signum, frame):
        logging.info('Received shutdown signal %s, initiating graceful shutdown...', signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_serve)
    signal.signal(signal.SIGTERM, stop_serve)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(5).wait()
        servicer.stop()
        
        if os.path.exists(status_file_path):
            try:
                os.remove(status_file_path)
            except OSError:
                pass
                
        logging.info('server stopped completely.')


if __name__ == '__main__':
    serve()

최종개선사항
✅ 단순 프로세스 교체 → Startup 검증 후 Atomic Commit → Reload 실패 시 기존 정상 프로세스 보존
✅ self.proc 오타 및 상태 불일치 → 명시적 Lifecycle State 관리 → STOPPING/DEAD 상태에서 잘못된 RPC 실행 차단
✅ stderr 격리 선언 → 실제 PIPE 반환 구조 → stderr 미소비에 따른 버퍼 포화·서브프로세스 데드락 위험은 반드시 제거 필요
✅ 요청별 임시 reader thread → 프로세스 단위 지속형 stdout reader → 타임아웃마다 누적되는 좀비 reader thread와 동시 readline() 경쟁 방지
✅ Lock 범위 최소화 → 프로세스 참조와 실제 stdin/stdout I/O의 동기화 구조 재설계 → Reload/Stop과 Input 간 프로세스 수명 경합 방지
✅ Reload 호출별 독립적인 Atomic Commit → Reload 전체를 직렬화하는 상태 전이 제어 → 동시 Reload에서 구형·신형 프로세스가 뒤섞이는 상태 오염 방지
✅ 명령어 일부 문자열만 마스킹 → 인자 단위 비밀정보 정책 및 stderr 정책 명확화 → 운영 로그·프로세스 오류 출력의 민감정보 노출 방지

원본의 치명적 생명주기 결함은 상당 부분 제거됐지만, 현재 버전은 아직 stdout 동시 읽기·reader thread 누적·stderr PIPE 미소비·동시 Reload 경합이라는 4개의 핵심 경쟁조건이 남아 있어 9.8급 운영 코드로 판정하기엔 이르다.
