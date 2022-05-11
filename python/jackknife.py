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


from typing import List, Union
import numpy as np
from dataset import Sample
from utils import uniform_resample, norm_comp_len, norm_diagonal
from stats import RecognitionResult
from tables import DeviceType, get_device_resample_count

# Implementation based on the paper:
# Taranta II, E. M., Samiei, A., Maghoumi, M., Khaloo, P., Pittman, C. R., 
# and LaViola Jr, J. J. " Jackknife: A reliable recognizer with few 
# samples and many modalities.", CHI 2017.
# 
# For more implentations, refer to: https://github.com/ISUE/Jackknife

class Jackknife(object):
    """Recognizer Class"""

    class Template(object):
        """Jackknife inner representation of a sample"""

        def __init__(self,
                     sample: Sample,
                     n: int):
            """ """
            self.gname = sample.gname
            self.sample = sample

            self.pts = uniform_resample(sample.trajectory, n)
            self.vecs = self.vectorize(self.pts)

            # Component-wise absolute distance (along each axis)
            self.abs = norm_comp_len(self.pts)

            # Normalized size of bounding box
            self.bb = norm_diagonal(self.pts)

        @staticmethod
        def vectorize(pts: Union[np.ndarray, list]):
            """Vectorize points"""

            vecs = []
            for i in range(1, len(pts)):
                vec = pts[i] - pts[i - 1]
                length = np.linalg.norm(vec)
                if length == 0.0:
                    vecs += [vec]
                    continue
                vecs += [vec / length]
            return vecs

    def __init__(self,
                 resample_cnt: int,
                 device_type: DeviceType,
                 window: float = .1):
        """Init Jakknife with resample count and percent DTW window width

        Args:
            resample_cnt (int): Number of points to resample each template to
            device_type (DeviceType): Device type (for synthetic data generation purposes)
            window (float, optional): Ratio of template length to use for DTW. Defaults to .1.
        """
        self.resample_cnt = resample_cnt
        self.window = int(round(resample_cnt * window))
        self.templates = []
        self.samples = []
        self.rejection_threshold = np.inf
        self.device_type = device_type

    def __iadd__(self, sample: Sample):
        """Add Jackknife template based on Sample"""
        template = Jackknife.Template(sample, self.resample_cnt)
        self.templates += [template]
        self.samples += [sample]
        return self

    def get_training_set(self) -> List[Sample]:
        return self.samples

    def add_template(self, sample: Sample) -> None:
        """Add Jackknife template based on Sample"""
        self.__iadd__(sample)

    def classify(self, candidate: Sample) -> List[RecognitionResult]:
        """Get list of matches sorted by best scores"""
        c = Jackknife.Template(candidate, self.resample_cnt)
        ret = []

        for t in self.templates:

            cf = 1.0

            abs_dot = np.dot(c.abs, t.abs)
            bb_dot = np.dot(c.bb, t.bb)

            cf *= 1.0 / max(0.01, abs_dot)
            cf *= 1.0 / max(0.01, bb_dot)

            score = cf

            score *= self.dtw_vec(
                c.vecs,
                t.vecs,
                self.window)

            if score < self.rejection_threshold:
                r = RecognitionResult(score, t)
                ret += [r]

        if len(ret) == 0:
            return None

        return sorted(ret)

    def measure(self, candidate_pts: Union[list, np.ndarray], tidx: int) -> float:
        """Return score against points"""

        candidate_pts = uniform_resample(candidate_pts, n=self.resample_cnt)
        c_vecs = Jackknife.Template.vectorize(candidate_pts)

        window = int(round(len(candidate_pts) * .1))

        score = Jackknife.dtw_vec(c_vecs,
                                  self.templates[tidx].vecs,
                                  window)

        return score

    @staticmethod
    def dtw_vec(cvecs: Union[list, np.ndarray],
                tvecs: Union[list, np.ndarray],
                window: int) -> float:
        """Calculate warping distance"""

        assert (len(cvecs) == len(tvecs))
        n = len(cvecs)

        dtw = np.full((n + 1, n + 1), np.inf)
        dtw[0, 0] = 0.0

        for ii in range(1, n + 1):
            minimum = max(1, ii - window)
            maximum = min(ii + window, n)

            for jj in range(minimum, maximum + 1):
                cost = 1 - np.dot(
                    cvecs[(ii - 1)],
                    tvecs[(jj - 1)])

                insert = dtw[ii - 1, jj]
                delete = dtw[ii, jj - 1]
                match = dtw[ii - 1, jj - 1]

                dtw[ii, jj] = min(insert, delete, match) + cost

        return dtw[n, n]

    def set_rejection_threshold(self, r: float):
        self.rejection_threshold = r
