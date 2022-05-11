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


import random
from typing import List, Union
import numpy as np

import utils

from dataset import Sample
from tables import get_gpsr_coefficients, get_inflation, DeviceType
from jackknife import Jackknife
from mincer import Mincer


def select_rejection_threshold(recognizer: Jackknife,
                               device_type: DeviceType,
                               beta: float,
                               train_cnt: int):
    """ """

    training_set = recognizer.get_training_set()
    template_cnt = len(training_set)

    # Set up a negative stream
    nstream = Mincer(recognizer)

    # Allocate distributions, one per template
    iteration_cnt = 10
    pos: List[float] = []
    neg: List[float] = []

    for tidx in range(template_cnt):

        # Generate negative samples
        for _ in range(iteration_cnt):
            minced_sample = nstream.mince(tidx)

            score = recognizer.measure(minced_sample, tidx)
            neg += [score]

        # Get GPSR parameters
        gpsr_r = 5
        gpsr_n = optimal_gpsr_n(training_set[tidx].trajectory,
                                device_type)

        # Generate positive samples
        for _ in range(iteration_cnt):
            positive_sample = gpsr(training_set[tidx].trajectory,
                                   n=gpsr_n,
                                   remove_cnt=gpsr_r,
                                   variance=0.25)

            score = recognizer.measure(positive_sample, tidx)
            pos += [score]

    print("mean pos score:", np.mean(pos, axis=0))
    print("mean neg score:", np.mean(neg, axis=0))

    threshold = estimate_rejection_threshold(pos, neg, beta)
    print("found thresh  :", threshold)
    inflation = get_inflation(device_type)
    scale = estimate_adjustment(6, threshold, inflation, train_cnt, beta)
    rejection = threshold * scale
    print("after scaling :", rejection)
    return rejection


def estimate_rejection_threshold(pos: List[float],
                                 neg: List[float],
                                 beta: Union[float, int]) -> float:
    """"""

    pos.sort()
    neg.sort()

    pcnt = len(pos)
    ncnt = len(neg)

    tp = 0
    fn = pcnt
    fp = 0
    best_threshold = -np.inf
    best_fscore = -np.inf

    pidx = 0
    nidx = 0

    b2 = beta * beta
    b2p1 = b2 + 1.0

    while pidx < pcnt and nidx < ncnt:
        threshold = min(pos[pidx], neg[nidx])

        while pidx < pcnt and pos[pidx] <= threshold:
            tp += 1
            fn -= 1
            pidx += 1

        while nidx < ncnt and neg[nidx] <= threshold:
            fp += 1
            nidx += 1

        # Estimate F-beta score and save if it's the best
        fscore = b2p1 * tp
        fscore /= ((b2p1 * tp) + (b2 * fn) + fp)

        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold

    return best_threshold


def optimal_gpsr_n(trajectory: List,
                   did: DeviceType):
    """ """

    resampled = utils.uniform_resample(trajectory, 64)

    minimum = np.min(resampled, axis=0)
    maximum = np.max(resampled, axis=0)

    prev = resampled[0]
    curr = resampled[0]

    angle = 0.0

    for i in range(1, len(resampled)):
        prev = curr
        curr = resampled[i] - resampled[i - 1]
        curr = utils.normalize_vector(curr)

        if i < 2:
            continue

        dot = max(-1.0, min(1.0, float(np.dot(prev, curr))))

        theta = np.arccos(dot)
        angle += theta

    # print(f"{angle=}")

    diagonal = np.linalg.norm(maximum - minimum)
    length = utils.path_length(trajectory)
    density = length / diagonal

    # print(f"{diagonal=}")
    # print(f"{length=}")
    # print(f"{density=}")
    # print()

    gpsr_co = get_gpsr_coefficients()

    ret = gpsr_co.intercept
    ret += gpsr_co.density_co * density
    ret += gpsr_co.angle_co * angle
    ret += gpsr_co.density_angle_co * density * angle

    return round(ret)



# Perform Gesture Path Stochastic Resampling (GPSR) to create a synthetic
# variation of the given trajectory:
#
# Eugene M. Taranta II, Mehran Maghoumi, Corey R. Pittman, Joseph J. LaViola Jr.
# "A Rapid Prototyping Approach to Synthetic Data Generation For Improved 2D 
# Gesture Recognition", UIST 2016


def gpsr(pts: Union[list, np.ndarray],
         n: int = 0,
         remove_cnt: int = 1,
         variance: float = 0.25):
    """ Gesture path stochastic resampling."""

    # first stochastically resample trajectory

    pts = utils.uniform_resample(pts, n + remove_cnt, variance)

    # then remove some random points
    for idx in range(remove_cnt):
        gotta_go = random.randint(0, len(pts) - 1)
        pts = np.delete(pts,
                        gotta_go,
                        axis=0)

    # And last, normalize and concatenate the
    # between point vectors.
    ret = np.empty((n, len(pts[0])))
    ret[0] = 0.0
    for idx in range(1, n):
        delta = pts[idx] - pts[idx - 1]
        ret[idx] = ret[idx - 1] + delta / np.linalg.norm(delta)

    return ret


def estimate_adjustment(dimension: float,
                        threshold: float,
                        inflation: float,
                        train_cnt: int,
                        beta: float, ) -> float:
    """"""
    rt = threshold  # radius training
    ri = threshold * inflation  # radius inflated
    best_fscore = 0
    adjustment = 0

    scale = 0.5
    while scale <= 2:
        tp = 0.0
        fp = 0.0
        fn = 0.0

        rmax = max(ri, rt + rt * scale)
        for ii in range(0, 1000):
            test = sample_sphere(rmax, dimension)
            inside = np.linalg.norm(test) <= ri
            recognized = False

            for train_no in range(train_cnt):
                tpt = sample_sphere(rt, dimension)
                l2 = np.linalg.norm(tpt - test)
                if l2 <= rt * scale:
                    recognized = True
                    break

            if inside and recognized:
                tp += 1
            if inside and not recognized:
                fn += 1
            if not inside and recognized:
                fp += 1

        fscore = calculate_fscore(beta, tp, fp, fn)

        if fscore > best_fscore:
            best_fscore = fscore
            adjustment = scale

        scale += 0.05

    return adjustment


def sample_sphere(radius: float,
                  dimension: float) -> np.ndarray:
    """"""
    pt = np.empty(int(dimension))

    while True:
        for ii in range(int(dimension)):
            x = float(random.uniform(0, 1))
            x = 2.0 * (x - 0.5) * radius
            pt[ii] = x

        l2 = np.linalg.norm(pt)
        if l2 <= radius:
            return pt


def calculate_fscore(beta: float, tp: float, fp: float, fn: float) -> float:
    b2 = beta * beta
    b2p1 = b2 + 1.0
    fscore = (b2p1 * tp)
    fscore /= max(((b2p1 * tp) + (b2 * fn) + fp), 1)
    return fscore
