# Copyright 2022 the University of Central Florida Research Foundation, Inc.
# All rights reserved.
#
#     Eugene M. Taranta II <etaranta@gmail.com>
#     Mykola Maslych <maslychm@knights.ucf.edu>
#     Ryan Ghamandi <ryanghamandi1@gmail.com>
#     Joseph J. LaViola Jr. <jjl@cs.ucf.edu>
#
# Subject to the terms and conditions of the Florida Public Educational
# Institution non-exclusive software license, this software is distributed
# under a non-exclusive, royalty-free, non-sublicensable, non-commercial,
# non-exclusive, academic research license, and is distributed without warranty
# of any kind express or implied.
#
# The Florida Public Educational Institution non-exclusive software license
# is located at <https://github.com/ISUE/VKM/blob/main/LICENSE>.


from typing import Union
import numpy as np
import random


def norm_diagonal(pts: list):
    pts = np.array(pts)
    minimum = np.min(pts, axis=0)
    maximum = np.max(pts, axis=0)
    delta = maximum - minimum
    return delta / np.linalg.norm(delta)


def diagonal(pts: list):
    pts = np.array(pts)
    minimum = np.min(pts, axis=0)
    maximum = np.max(pts, axis=0)
    return maximum - minimum


def norm_comp_len(pts):
    """Component-wise absolute distance"""
    pts = np.array(pts)
    abs_comp_len = np.full(pts.shape[1], 0)
    for p in pts:
        for idx, m in enumerate(p):
            abs_comp_len[idx] += abs(m)
    return abs_comp_len / np.linalg.norm(abs_comp_len)


def path_length(pts):
    """ """
    cnt = len(pts)
    ret = 0.0
    for idx in range(1, cnt):
        ret += np.linalg.norm(pts[idx] - pts[idx - 1])

    return ret


def uniform_resample(pts: Union[list, np.ndarray],
                     n: int,
                     variance: float = 0.0) -> np.ndarray:
    """ """
    #
    # create random intervals
    #
    scale = (12 * variance) ** .5
    intervals = [1.0 + random.uniform(0, 1) * scale for ii in range(n - 1)]
    total = sum(intervals)
    intervals = [val / total for val in intervals]

    # Setup place to store resampled points, and
    # store first point. jj indexes the return matrix
    ret = np.empty((n, len(pts[0])))
    ret[0] = pts[0]
    path_distance = path_length(pts)
    jj = 1

    # now do resampling
    accumulated_distance = 0.0
    interval = path_distance * intervals[jj - 1]

    for ii in range(1, len(pts)):

        distance = np.linalg.norm(pts[ii] - pts[ii - 1])

        if accumulated_distance + distance < interval:
            accumulated_distance += distance
            continue

        previous = pts[ii - 1]
        while accumulated_distance + distance >= interval:

            # Now we need to interpolate between the last point
            # and the current point.
            remaining = interval - accumulated_distance
            t = remaining / distance

            # Handle any precision errors. Note that the distance can
            # be zero if two samples are sufficiently close together,
            # which can result in nan.
            t = min(max(t, 0.0), 1.0)
            if not np.isfinite(t):
                t = 0.5

            ret[jj] = (1.0 - t) * previous + t * pts[ii]

            # Reduce the distances based on how much path we
            # just consumed.
            distance = distance - remaining
            accumulated_distance = 0.0
            previous = ret[jj]
            jj += 1

            # Exit early so we don't go past end
            # of the intervals array.
            if jj == n:
                break

            # select next interval
            interval = path_distance * intervals[jj - 1]

        accumulated_distance = distance

    if jj < n:
        ret[n - 1] = pts[ii - 1]
        jj += 1

    assert jj == n
    return ret


def normalize_vector(vec: list):
    return vec / np.linalg.norm(vec)


def get_cma_r(fs, fc):
    """"""
    m = 3.0 / 2.0 * (fs / (2.0 * np.pi * fc)) ** 2.0
    # print("m=", m)
    return int(np.ceil(m))


def rcma(pts, w, r=1):
    n = len(pts)
    for _ in range(r):
        ret = []
        for ii in range(n):
            tot = 0.0
            cnt = 0.0
            for jj in range(-w, w + 1):
                if ii + jj < 0:
                    continue
                if ii + jj >= n:
                    continue
                tot += pts[ii + jj]
                cnt += 1.0
            ret += [tot / cnt]
        pts = np.array(ret)
    return pts


def zero_vector(dimensions: int):
    return np.zeros(dimensions)


def minimum_point(vecs):
    return np.min(vecs, axis=0)


def maximum_point(vecs):
    return np.max(vecs, axis=0)


def maximum_component(vec):
    return np.amax(vec)


def distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)


def dot(v1, v2):
    return np.dot(v1, v2)


def arccos(angle):
    return np.arccos(angle)
