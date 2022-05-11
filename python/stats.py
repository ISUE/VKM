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


class RecognitionResult(object):
    """ """

    def __init__(self,
                 score,
                 template):
        self.score: float = score
        self.sample = template.sample
        self.gname: str = template.gname
        self.template = template

    def __lt__(self,
               other):
        return self.score < other.score


class InnerConfusionMatrix(object):
    """Per-gesture confusion matrix"""

    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def accuracy(self):
        """ """
        denom = self.tp + self.tn + self.fp + self.fn
        if denom == 0.0:
            return 0.0
        return (self.tp + self.tn) / denom

    def precision(self):
        """ """
        denom = self.tp + self.fp
        if denom == 0.0:
            return 0.0
        return self.tp / denom

    def recall(self):
        """ """
        denom = self.tp + self.fn
        if denom == 0.0:
            return 0.0
        return self.tp / denom

    def specificity(self):
        """ """
        denom = self.tn + self.fp
        if denom == 0.0:
            return 0.0
        return self.tn / denom

    def fscore(self):
        """ """
        denom = self.precision() + self.recall()
        if denom == 0.0:
            return 0.0
        numer = 2.0 * self.precision() * self.recall()
        return numer / denom

    def reset(self):
        """ """
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0


class ConfusionMatrix(object):
    """Macro confusion matrix"""

    def __init__(self, gnames):
        """Create inner matrices for each gname"""
        self.inner_confusion_matrices = {}
        for gname in gnames:
            self.inner_confusion_matrices[gname] = InnerConfusionMatrix()

    def update(self,
               expected: Union[str, int, None],
               classified: Union[str, int, None]):
        """ """

        if classified is None:
            expected_matrix = self.inner_confusion_matrices[expected]
            expected_matrix.fn += 1.0
            return

        if expected == classified:
            expected_matrix = self.inner_confusion_matrices[expected]
            expected_matrix.tp += 1.0
            return

        if expected != classified:
            expected_matrix = self.inner_confusion_matrices[expected]
            classified_matrix = self.inner_confusion_matrices[classified]
            expected_matrix.fn += 1.0
            classified_matrix.fp += 1.0
            return

    def fscore(self) -> float:
        ret = 0.0
        matrices_cnt = len(self.inner_confusion_matrices)
        for gname, m in self.inner_confusion_matrices.items():
            ret += m.fscore()

        return ret / matrices_cnt
