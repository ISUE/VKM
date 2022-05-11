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


from dataset import Sample
from jackknife import Jackknife
import utils
from tables import DeviceType

import numpy as np
import random


class Mincer:
    def __init__(self, recognizer: Jackknife):
        self.device_type: DeviceType = recognizer.device_type
        self.training_samples = recognizer.get_training_set()
        self.prepared_trajectories = [
            Mincer.prepare_trajectory(t) for t in self.training_samples]

    @staticmethod
    def prepare_trajectory(sample: Sample):
        """"""
        prepared = utils.uniform_resample(sample.trajectory, n=128)

        # move origin to centroid and scale longer side to 1
        centroid = np.mean(prepared, axis=0)
        width = np.max(prepared, axis=0) - np.min(prepared, axis=0)
        scale = np.amax(width)
        for pt in prepared:
            pt -= centroid
            pt /= scale

        vecs = []
        for i in range(1, len(prepared)):
            vecs += [prepared[i] - prepared[i - 1]]
        return vecs

    def mince(self, target_idx: int):
        """"""
        gesture_class_name = self.training_samples[target_idx].gname
        cnt = len(self.prepared_trajectories)

        # Select random template sample of different gesture
        other_idx = random.randint(0, cnt - 1)
        while gesture_class_name == self.training_samples[other_idx].gname:
            other_idx = random.randint(0, len(self.prepared_trajectories) - 1)

        n = len(self.prepared_trajectories[0])
        threshold = n / 3

        # Select indices for splicing
        while True:
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            width = abs(idx2 - idx1) + 1

            if width < threshold:
                continue
            break

        m = len(self.prepared_trajectories[0][0])
        pt = np.zeros(m)

        minced_sample = [pt]

        # Copy the first part of trajectory
        for i in range(min(idx1, idx2)):
            pt = pt + self.prepared_trajectories[target_idx][i]
            minced_sample += [pt]

        # Copy the middle part of trajectory
        ii = idx1
        if ii <= idx2:
            while ii <= idx2:
                pt = pt + self.prepared_trajectories[other_idx][ii]
                minced_sample += [pt]
                ii += 1
        else:
            while ii >= idx2:
                pt = pt + self.prepared_trajectories[other_idx][ii]
                minced_sample += [pt]
                ii -= 1

        # Copy the last part of trajectory
        for i in range(max(idx1, idx2) + 1, n):
            pt = pt + self.prepared_trajectories[target_idx][i]
            minced_sample += [pt]

        return minced_sample
