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


import os

from dataset import Dataset
from jackknife import Jackknife
from download_ha import dowload_ha
from stats import ConfusionMatrix
import synthetic
import tables
from tables import DeviceType
from numpy import mean


def run_recognizer(dataset: Dataset,
                   train_count: int,
                   iteration_count: int):
    """Run recognizer with specified parameters"""

    ds_iterator = dataset.ud(train_count, iteration_count)
    resample_count = tables.get_device_resample_count(dataset.device_type)
    cfm = ConfusionMatrix(dataset.gnames)
    scores = []

    correct, total = 0, 0
    for idx, (train, test) in enumerate(ds_iterator):
        print(
            f"Iteration: {idx + 1} / {iteration_count * len(dataset.snames)}")

        # Create recognizer
        recognizer = Jackknife(resample_cnt=resample_count,
                               device_type=dataset.device_type)

        # Train recognizer
        for t in train:
            recognizer.add_template(t)

        # Determine and set the rejection threshold
        thresh = synthetic.select_rejection_threshold(
            recognizer,
            device_type,
            beta=1,
            train_cnt=train_count,
        )

        recognizer.set_rejection_threshold(thresh)

        for t in test:
            results = recognizer.classify(t)

            classified_gname = None
            if results is not None:
                classified_gname = results[0].gname

            cfm.update(t.gname, classified_gname)

            correct += float(t.gname == classified_gname)
            if t.gname == classified_gname:
                scores += [results[0].score]

            total += 1.0

        print()

    print("accuracy: {:2.2f}".format(float(correct / total)))
    print("F score:", cfm.fscore())
    # print("avg winning score: ", mean(scores))

    return float(correct / total), cfm.fscore()


if __name__ == '__main__':
    """ """
    device_type = DeviceType.KINECT
    use_filter = True
    device_path = tables.get_device_dataset_path(device_type)

    if not os.path.isdir(device_path):
        print("Dataset not found. Downloading...")
        dowload_ha("../")

    ds = Dataset.Load(device_path,
                      device_type,
                      filter=use_filter
                      )

    # print(ds)

    run_recognizer(dataset=ds,
                   train_count=1,
                   iteration_count=1)
