원본코드
# PageRank Python implementation by Vincent Kraeutler, at:
# http://kraeutler.net/vincent/essays/google%20page%20rank%20in%20python
#
# The code has the following changes from the original:
#
# * The data types were converted from 32 bits to 64 bits and precision
#
# * The convergence criterion was changed from the average deviation
# to the Euclidean 1-norm distance, which is tighter
# 
# The original code was released by Vincent Kraeutler under a Creative
# Commons Attribution 2.5 License
#

#!/usr/bin/env python

from numpy import *

def pageRankGenerator(At = [array((), int64)], 
                      numLinks = array((), int64),  
                      ln = array((), int64),
                      alpha = 0.85, 
                      convergence = 0.0001, 
                      checkSteps = 10
                      ):
    """
    Compute an approximate page rank vector of N pages to within
    some convergence factor.

    @param At a sparse square matrix with N rows. At[ii] contains
    the indices of pages jj linking to ii.

    @param numLinks iNumLinks[ii] is the number of links going out
    from ii.

    @param ln contains the indices of pages without links

    @param alpha a value between 0 and 1. Determines the relative
    importance of "stochastic" links.

    @param convergence a relative convergence criterion. Smaller
    means better, but more expensive.

    @param checkSteps check for convergence after so many steps
    """

    # the number of "pages"
    N = len(At)

    # the number of "pages without links"
    M = ln.shape[0]

    # initialize: single-precision should be good enough
    iNew = ones((N,), float64) / N
    iOld = ones((N,), float64) / N

    done = False
    while not done:

        # normalize every now and then for numerical stability
        iNew /= sum(iNew)

        for step in range(checkSteps):

            # swap arrays
            iOld, iNew = iNew, iOld

            # an element in the 1 x I vector. 
            # all elements are identical.
            oneIv = (1 - alpha) * sum(iOld) / N

            # an element of the A x I vector.
            # all elements are identical.
            oneAv = 0.0
            if M > 0:
                oneAv = alpha * sum(iOld.take(ln, axis = 0)) / N

            # the elements of the H x I multiplication
            ii = 0 
            while ii < N:
                page = At[ii]
                h = 0
                if page.shape[0]:
                    h = alpha * dot(
                            iOld.take(page, axis = 0),
                            1. / numLinks.take(page, axis = 0)
                            )
                iNew[ii] = h + oneAv + oneIv
                ii += 1

        diff = sum(abs(iNew - iOld))
        done = (diff < convergence)

        yield iNew


def transposeLinkMatrix(
        outGoingLinks = [[]]
        ):
    """
    Transpose the link matrix. The link matrix contains the pages
    each page points to. But what we want is to know which pages
    point to a given page, while retaining information about how
    many links each page contains (so store that in a separate
    array), as well as which pages contain no links at all (leaf
    nodes).

    @param outGoingLinks outGoingLinks[ii] contains the indices of
    pages pointed to by page ii

    @return a tuple of (incomingLinks, numOutGoingLinks, leafNodes)
    """

    nPages = len(outGoingLinks)
    # incomingLinks[ii] will contain the indices jj of the pages
    # linking to page ii
    incomingLinks = [[] for ii in range(nPages)]
    # the number of links in each page
    numLinks = zeros(nPages, int64)
    # the indices of the leaf nodes
    leafNodes = []

    for ii in range(nPages):
        if len(outGoingLinks[ii]) == 0:
            leafNodes.append(ii)
        else:
            numLinks[ii] = len(outGoingLinks[ii])
            # transpose the link matrix
            for jj in outGoingLinks[ii]:
                incomingLinks[jj].append(ii)

    incomingLinks = [array(ii) for ii in incomingLinks]
    numLinks = array(numLinks)
    leafNodes = array(leafNodes)

    return incomingLinks, numLinks, leafNodes


def pageRank(
        linkMatrix = [[]],
        alpha = 0.85, 
        convergence = 0.0001, 
        checkSteps = 10
        ):
    """
    Convenience wrap for the link matrix transpose and the generator.

    @see pageRankGenerator for parameter description
    """
    incomingLinks, numLinks, leafNodes = transposeLinkMatrix(linkMatrix)

    for gr in pageRankGenerator(incomingLinks, numLinks, leafNodes,
                                alpha = alpha, convergence = convergence,
                                checkSteps = checkSteps):
        final = gr

    return final

수학적 핵심은 정확하지만, 무방비 입력·무한루프 가능성·Python 순차 루프·불필요한 제너레이터 구조 때문에 대규모 프로덕션 환경에서는 취약하다.
특히 그래프 무결성 검증과 종료 보장을 먼저 확보한 뒤, 병목인 노드별 순차 연산을 희소/벡터화 구조로 전환하는 것이 핵심 개선점이다.

제안패치
from __future__ import absolute_import, division, print_function

import numpy as np


def validate_pagerank_inputs(
    at,
    num_links,
    ln,
    alpha,
    convergence,
    check_steps,
    max_iterations,
):
    if not isinstance(at, (list, tuple)):
        raise TypeError("at must be a list or tuple.")

    if type(alpha) not in (int, float) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")

    if type(convergence) not in (int, float) or convergence <= 0.0:
        raise ValueError("convergence must be positive.")

    if type(check_steps) is not int or check_steps <= 0:
        raise ValueError("check_steps must be a positive integer.")

    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")

    n_pages = len(at)

    if n_pages == 0:
        raise ValueError("Graph cannot be empty.")

    num_links = np.asarray(num_links)
    ln = np.asarray(ln)

    if num_links.ndim != 1 or len(num_links) != n_pages:
        raise ValueError(
            "num_links must be a 1-D array with one entry per page."
        )

    if ln.ndim != 1:
        raise ValueError("ln must be a 1-D array.")

    if np.any(num_links < 0):
        raise ValueError("num_links cannot contain negative values.")

    if ln.size:
        if np.any(ln < 0) or np.any(ln >= n_pages):
            raise ValueError("Leaf-node indices are out of range.")

    for page in at:
        page = np.asarray(page)
        if page.ndim != 1:
            raise ValueError("Each incoming-link entry must be 1-D.")

        if page.size:
            if np.any(page < 0) or np.any(page >= n_pages):
                raise ValueError("Incoming-link index is out of range.")

            if np.any(num_links[page] <= 0):
                raise ValueError(
                    "Incoming links reference a page with zero out-degree."
                )

    return n_pages, num_links.astype(np.float64, copy=False), ln.astype(
        np.int64, copy=False
    )


def page_rank_generator(
    at=None,
    num_links=None,
    ln=None,
    alpha=0.85,
    convergence=0.0001,
    check_steps=10,
    max_iterations=10000,
):
    at = [] if at is None else at
    num_links = (
        np.array([], dtype=np.int64)
        if num_links is None
        else num_links
    )
    ln = (
        np.array([], dtype=np.int64)
        if ln is None
        else ln
    )

    (
        n_pages,
        num_links,
        ln,
    ) = validate_pagerank_inputs(
        at,
        num_links,
        ln,
        alpha,
        convergence,
        check_steps,
        max_iterations,
    )

    i_new = np.full(n_pages, 1.0 / n_pages, dtype=np.float64)
    i_old = np.empty_like(i_new)

    for iteration in range(max_iterations):
        i_new /= np.sum(i_new)

        for _ in range(check_steps):
            i_old, i_new = i_new, i_old

            dangling_mass = 0.0
            if ln.size:
                dangling_mass = alpha * np.sum(i_old[ln]) / n_pages

            teleport_mass = (1.0 - alpha) / n_pages

            for page_index, incoming in enumerate(at):
                if incoming.size:
                    contribution = np.sum(
                        i_old[incoming] / num_links[incoming]
                    )
                    i_new[page_index] = (
                        alpha * contribution
                        + dangling_mass
                        + teleport_mass
                    )
                else:
                    i_new[page_index] = (
                        dangling_mass + teleport_mass
                    )

        diff = np.sum(np.abs(i_new - i_old))

        yield i_new.copy()

        if diff < convergence:
            return

    raise RuntimeError(
        "PageRank failed to converge within "
        f"{max_iterations} iterations."
    )


def transpose_link_matrix(out_going_links=None):
    if out_going_links is None:
        raise ValueError("out_going_links must be provided.")

    if not isinstance(out_going_links, (list, tuple)):
        raise TypeError("out_going_links must be a list or tuple.")

    n_pages = len(out_going_links)

    if n_pages == 0:
        raise ValueError("Graph cannot be empty.")

    incoming_links = [[] for _ in range(n_pages)]
    num_links = np.zeros(n_pages, dtype=np.int64)
    leaf_nodes = []

    for source, links in enumerate(out_going_links):
        if links is None:
            raise TypeError(f"Links for page {source} cannot be None.")

        if len(links) == 0:
            leaf_nodes.append(source)
            continue

        num_links[source] = len(links)

        for target in links:
            if type(target) is not int:
                raise TypeError(
                    f"Link target {target!r} must be an integer."
                )

            if target < 0 or target >= n_pages:
                raise ValueError(
                    f"Link target {target} is outside graph bounds."
                )

            incoming_links[target].append(source)

    return (
        [np.asarray(links, dtype=np.int64) for links in incoming_links],
        num_links,
        np.asarray(leaf_nodes, dtype=np.int64),
    )


def page_rank(
    link_matrix=None,
    alpha=0.85,
    convergence=0.0001,
    check_steps=10,
    max_iterations=10000,
):
    if link_matrix is None:
        raise ValueError("link_matrix must be provided.")

    incoming_links, num_links, leaf_nodes = transpose_link_matrix(
        link_matrix
    )

    final = None

    for result in page_rank_generator(
        incoming_links,
        num_links,
        leaf_nodes,
        alpha=alpha,
        convergence=convergence,
        check_steps=check_steps,
        max_iterations=max_iterations,
    ):
        final = result

    if final is None:
        raise RuntimeError("PageRank produced no result.")

    return final

최종 개선사항
✅ 무제한 while 수렴 루프 → max_iterations 기반 종료 보장 → 무한 실행에 의한 엔진 고착 방지
✅ 불완전한 파라미터 검증 → 그래프 인덱스·배열 길이·zero-degree 계약 검증 → 입력 오염 조기 차단
✅ yield i_new mutable buffer 노출 → i_new.copy() 반환 → iteration 결과 무결성 보장
✅ while 제거를 벡터화로 오인한 구조 → 실제 Python 노드 순회 병목 명시 → 성능 개선 범위 정확화
✅ mutable default argument → None 기반 안전한 초기화 → 호출 간 상태 오염 방지
✅ 수렴 실패를 무기한 대기 → 명시적 RuntimeError 종료 → 배치·서비스 엔진 셧다운 방지

원본의 PageRank 수학적 구조는 유지하면서 입력 무결성·종료 안정성·결과 독립성을 강화했으며, 진정한 대규모 성능 승격은 CSR 기반 희소 행렬 연산으로 넘어가야 완성된다.
