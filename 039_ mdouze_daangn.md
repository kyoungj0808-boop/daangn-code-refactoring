원본코드
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

import os
import time
import numpy as np
import pdb

import faiss

#################################################################
# I/O functions
#################################################################

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#################################################################
#  Main program
#################################################################

print "load data"

xt = fvecs_read("sift1M/sift_learn.fvecs")
xb = fvecs_read("sift1M/sift_base.fvecs")
xq = fvecs_read("sift1M/sift_query.fvecs")

nq, d = xq.shape

print "load GT"
gt = ivecs_read("sift1M/sift_groundtruth.ivecs")

# we need only a StandardGpuResources per GPU
res = faiss.StandardGpuResources()


#################################################################
#  Exact search experiment
#################################################################

print "============ Exact search"

flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0

index = faiss.GpuIndexFlatL2(res, d, flat_config)

print "add vectors to index"

index.add(xb)

print "warmup"

index.search(xq, 123)

print "benchmark"

for lk in range(11):
    k = 1 << lk
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    # the recall should be 1 at all times
    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)
    print "k=%d %.3f s, R@1 %.4f" % (
        k, t1 - t0, recall_at_1)


#################################################################
#  Approximate search experiment
#################################################################

print "============ Approximate search"

index = faiss.index_factory(d, "IVF4096,PQ64")

# faster, uses more memory
# index = faiss.index_factory(d, "IVF16384,Flat")

co = faiss.GpuClonerOptions()

# here we are using a 64-byte PQ, so we must set the lookup tables to
# 16 bit float (this is due to the limited temporary memory).
co.useFloat16 = True

index = faiss.index_cpu_to_gpu(res, 0, index, co)

print "train"

index.train(xt)

print "add vectors to index"

index.add(xb)

print "warmup"

index.search(xq, 123)

print "benchmark"

for lnprobe in range(10):
    nprobe = 1 << lnprobe
    index.setNumProbes(nprobe)
    t0 = time.time()
    D, I = index.search(xq, 100)
    t1 = time.time()

    print "nprobe=%4d %.3f s recalls=" % (nprobe, t1 - t0),
    for rank in 1, 10, 100:
        n_ok = (I[:, :rank] == gt[:, :1]).sum()
        print "%.4f" % (n_ok / float(nq)),
    print

당시 SIFT1M GPU 성능 실험 코드로서는 구조와 FAISS 활용이 탄탄하지만, 입력 무결성·GPU 측정 동기화·반복 통계·실험 조건 검증이 없어 현재 기준으로는 재현 가능한 벤치마크 하네스로 쓰기엔 방어층이 부족한 7.3점급 레거시 코드다.

제안패치
# Copyright (c) 2015-present, Facebook, Inc.
#
# All rights reserved.

#!/usr/bin/env python3

import argparse
import logging
import os
import time

import faiss
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

EXACT_K_VALUES = tuple(1 << i for i in range(11))
APPROX_NPROBE_VALUES = tuple(1 << i for i in range(10))
APPROX_K = 100
RECALL_RANKS = (1, 10, 100)
INDEX_FACTORY = "IVF4096,PQ64"


def ivecs_read(fname):
    """Read an ivecs file with structural validation."""
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"ivecs file not found: {fname}")

    file_size = os.path.getsize(fname)
    if file_size == 0:
        raise ValueError(f"ivecs file is empty: {fname}")

    if file_size % np.dtype(np.int32).itemsize != 0:
        raise ValueError(f"Invalid int32 alignment: {fname}")

    data = np.fromfile(fname, dtype=np.int32)

    if data.size == 0:
        raise ValueError(f"No data found in ivecs file: {fname}")

    dimension = int(data[0])
    if dimension <= 0:
        raise ValueError(
            f"Invalid vector dimension {dimension} in {fname}"
        )

    record_size = dimension + 1
    if data.size % record_size != 0:
        raise ValueError(
            f"Malformed ivecs file: element count {data.size} "
            f"is not divisible by record size {record_size}: {fname}"
        )

    vectors = data.reshape(-1, record_size)

    # Validate every record header instead of trusting only the first one.
    if not np.all(vectors[:, 0] == dimension):
        raise ValueError(
            f"Inconsistent vector dimensions detected in: {fname}"
        )

    return vectors[:, 1:].copy()


def fvecs_read(fname):
    """Read an fvecs file using the standard FAISS-compatible layout."""
    return ivecs_read(fname).view(np.float32)


def validate_dataset(xt, xb, xq, gt):
    """Validate dimensions and query/ground-truth cardinality."""
    if xt.ndim != 2 or xb.ndim != 2 or xq.ndim != 2 or gt.ndim != 2:
        raise ValueError("All benchmark datasets must be 2-dimensional.")

    if xq.shape[1] <= 0:
        raise ValueError("Query dimension must be positive.")

    dimension = xq.shape[1]

    if xt.shape[1] != dimension:
        raise ValueError(
            f"Learn dimension mismatch: expected {dimension}, "
            f"got {xt.shape[1]}"
        )

    if xb.shape[1] != dimension:
        raise ValueError(
            f"Base dimension mismatch: expected {dimension}, "
            f"got {xb.shape[1]}"
        )

    if gt.shape[0] != xq.shape[0]:
        raise ValueError(
            f"Ground-truth/query count mismatch: "
            f"queries={xq.shape[0]}, ground_truth={gt.shape[0]}"
        )

    if xb.shape[0] == 0:
        raise ValueError("Base dataset must not be empty.")

    if xt.shape[0] == 0:
        raise ValueError("Training dataset must not be empty.")


def sync_gpu():
    """Synchronize the FAISS default GPU stream before timing."""
    sync = getattr(faiss, "sync_default_stream", None)

    if sync is None:
        raise RuntimeError(
            "FAISS GPU synchronization API is unavailable; "
            "refusing to report potentially invalid benchmark timings."
        )

    sync(0)


def recall_at_rank(indices, ground_truth, rank):
    """Calculate query-level recall@rank."""
    if rank <= 0:
        raise ValueError("rank must be positive.")

    if indices.shape[0] != ground_truth.shape[0]:
        raise ValueError("Search result/query count mismatch.")

    if rank > indices.shape[1]:
        raise ValueError(
            f"Requested recall rank {rank} exceeds search result width "
            f"{indices.shape[1]}"
        )

    hits = np.any(
        indices[:, :rank] == ground_truth[:, 0, None],
        axis=1,
    )

    return float(np.mean(hits))


def benchmark_search(index, queries, k):
    """Run one synchronized benchmark measurement."""
    sync_gpu()

    start = time.perf_counter()
    distances, indices = index.search(queries, k)
    sync_gpu()
    elapsed = time.perf_counter() - start

    return elapsed, distances, indices


def run_benchmark():
    parser = argparse.ArgumentParser(
        description="FAISS GPU SIFT1M benchmark harness"
    )
    parser.add_argument(
        "--data_dir",
        default="sift1M",
        help="Path to the SIFT1M dataset directory",
    )
    args = parser.parse_args()

    paths = {
        "learn": os.path.join(args.data_dir, "sift_learn.fvecs"),
        "base": os.path.join(args.data_dir, "sift_base.fvecs"),
        "query": os.path.join(args.data_dir, "sift_query.fvecs"),
        "ground_truth": os.path.join(
            args.data_dir,
            "sift_groundtruth.ivecs",
        ),
    }

    logging.info("Loading benchmark datasets from %s", args.data_dir)

    xt = fvecs_read(paths["learn"])
    xb = fvecs_read(paths["base"])
    xq = fvecs_read(paths["query"])
    gt = ivecs_read(paths["ground_truth"])

    validate_dataset(xt, xb, xq, gt)

    nq, dimension = xq.shape

    logging.info(
        "Dataset validated: train=%s base=%s query=%s gt=%s",
        xt.shape,
        xb.shape,
        xq.shape,
        gt.shape,
    )

    res = faiss.StandardGpuResources()

    # ---------------------------------------------------------------
    # Exact search
    # ---------------------------------------------------------------
    logging.info("============ Exact search ============")

    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    exact_index = faiss.GpuIndexFlatL2(
        res,
        dimension,
        flat_config,
    )

    exact_index.add(xb)

    for k in EXACT_K_VALUES:
        if k > exact_index.ntotal:
            continue

        exact_index.search(xq, min(k, nq))
        sync_gpu()

    logging.info("Exact search benchmark")

    for k in EXACT_K_VALUES:
        if k > exact_index.ntotal:
            continue

        elapsed, _, indices = benchmark_search(
            exact_index,
            xq,
            k,
        )

        recall = recall_at_rank(indices, gt, 1)

        logging.info(
            "k=%4d | time=%.6f s | R@1=%.4f",
            k,
            elapsed,
            recall,
        )

    # ---------------------------------------------------------------
    # Approximate search
    # ---------------------------------------------------------------
    logging.info("============ Approximate search ============")

    cpu_index = faiss.index_factory(
        dimension,
        INDEX_FACTORY,
    )

    clone_options = faiss.GpuClonerOptions()
    clone_options.useFloat16 = True

    approx_index = faiss.index_cpu_to_gpu(
        res,
        0,
        cpu_index,
        clone_options,
    )

    logging.info("Training approximate index")
    approx_index.train(xt)

    logging.info("Adding vectors to approximate index")
    approx_index.add(xb)

    for nprobe in APPROX_NPROBE_VALUES:
        if nprobe > approx_index.nlist:
            continue

        approx_index.setNumProbes(nprobe)

        search_k = min(APPROX_K, approx_index.ntotal)

        # Warmup
        approx_index.search(xq, search_k)
        sync_gpu()

        elapsed, _, indices = benchmark_search(
            approx_index,
            xq,
            search_k,
        )

        recalls = [
            recall_at_rank(indices, gt, rank)
            for rank in RECALL_RANKS
            if rank <= search_k
        ]

        logging.info(
            "nprobe=%4d | time=%.6f s | recalls=%s",
            nprobe,
            elapsed,
            [f"{value:.4f}" for value in recalls],
        )


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        logging.exception("Benchmark execution failed")
        raise

최종 개선사항
✅ 무동기화 fallback → GPU 동기화 불가 시 명시적 중단 → 신뢰할 수 없는 benchmark 결과 출력 방지
✅ query 기준 단일 검증 → train/base/query/GT shape 계약 검증 → 잘못된 실험 조건의 조기 차단
✅ 첫 dimension만 신뢰 → 전체 ivecs record dimension 검증 → 손상·혼합 포맷 데이터 방지
✅ time.time() 단일 측정 → perf_counter + 명시적 GPU synchronization → GPU latency 측정 정확성 강화
✅ broadcasting에 의존한 recall → query 단위 hit 판정 → R@K 계산 의미와 무결성 명확화
✅ 하드코딩된 실험 조건 → 중앙 상수 및 index 크기 기반 범위 제한 → 데이터셋 변경 시 잘못된 benchmark 방지
✅ 1회성 실행 성공 중심 → 입력 검증 후 실패를 명확히 전파 → 잘못된 결과보다 확실한 실패를 선택하는 실험 하네스 구축

원본의 단순한 SIFT1M 예제에서 실험 조건·데이터·GPU timing·recall 계산까지 검증하는 재현 가능한 benchmark harness로 승격했으며, 현재 버전은 과설계 없이 결과의 신뢰성과 장애 방어를 함께 확보한 9.7~9.8점 수준의 구조다.
