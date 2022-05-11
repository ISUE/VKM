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


from typing import Tuple
import numpy as np
import os
import struct
from random import sample as RS

from tables import DeviceType
import utils


class Dataset(object):
    """ """

    def __init__(
            self,
            samples,
            device_type: DeviceType):
        """ """
        gestures = {}
        subjects = {}
        sgestures = {}

        self.device_type = device_type

        for sample in samples:

            sname = sample.sname
            if sname not in subjects:
                subjects[sname] = []
                sgestures[sname] = {}
            subjects[sname] += [sample]

            gname = sample.gname
            if gname not in gestures:
                gestures[gname] = []
            gestures[gname] += [sample]

            if gname not in sgestures[sname]:
                sgestures[sname][gname] = []
            sgestures[sname][gname] += [sample]

        self.subjects = subjects
        self.gestures = gestures
        self.sgestures = sgestures
        self.samples = samples
        self.gnames = sorted(gestures.keys())
        self.snames = sorted(subjects.keys())

    def __str__(self):
        """ """
        snames = self.snames
        gnames = self.gnames
        samples = self.samples
        ret = "Subject Cnt: {}\n".format(len(snames))
        ret += "Gesture Cnt: {}\n".format(len(gnames))
        ret += "Sample Cnt:  {}\n".format(len(samples))
        length = max([len(gname) for gname in gnames])
        fmt = "\t{{:{}}} {{}}\n".format(length)

        for gname in gnames:
            ret += fmt.format(
                gname,
                len(self.gestures[gname]))
        return ret

    def __iter__(self):
        """ """
        return iter(self.gnames)

    def __getitem__(self, key):
        """ """
        if type(key) is str:
            if key in self.gnames:
                return self.gestures[key]
        raise KeyError("Unable to find {}".format(key))

    @property
    def fps(self):
        """ """
        return np.mean([s.fps for s in self.samples])

    @classmethod
    def Path(
            cls,
            start,
            name='datasets'):
        """ """
        path = os.path.dirname(os.path.realpath(start))
        while True:
            listing = os.listdir(path)
            if name in listing:
                return path + "/" + name
            dirname = os.path.dirname(path)
            if path == dirname:
                break
            path = dirname

    @classmethod
    def Load(
            cls,
            path: str,
            device_type: DeviceType,
            criteria={},
            filter: bool = False):
        """ """
        #
        # Get path to dataset if not specified.
        #
        if not os.path.exists(path):
            tmp = Dataset.Path(".")
            # print(tmp)
            assert tmp is not None
            path = tmp + '/' + path
        #
        # Validate input.
        #
        assert os.path.exists(path)
        assert os.path.isdir(path)

        #
        # Setup criteria...
        # Note, 'g'=gesture and 's'=subject
        #
        gexclude = criteria.get('gexclude', [])
        ginclude = criteria.get('ginclude', [])
        sexclude = criteria.get('sexclude', [])
        sinclude = criteria.get('sinclude', [])

        pid = 0
        samples = []
        for sname in os.listdir(path):

            spath = path + "/" + sname
            if not os.path.isdir(spath):
                continue

            if sname in sexclude:
                continue

            if len(sinclude) and sname not in sinclude:
                continue

            print("Load", sname)

            for gname in os.listdir(spath):

                gpath = spath + "/" + gname
                if gname in gexclude:
                    continue

                if len(ginclude) and gname not in ginclude:
                    continue

                if not os.path.isdir(gpath):
                    continue

                for ename in os.listdir(gpath):
                    epath = gpath + "/" + ename
                    sample = Sample.Load(
                        sname,
                        gname,
                        ename,
                        epath,
                        pid)

                    samples += [sample]

            pid += 1

        ret = cls(samples, device_type)
        ret.path = path

        if filter:
            ret.run_filter()

        return ret

    def ud(
            self,
            train_cnt,
            iterations):
        """ """
        for sname1 in self.snames:
            for _ in range(iterations):
                train = []
                test = []
                for gname, samples in self.sgestures[sname1].items():
                    samples = RS(samples, train_cnt + 1)
                    train += samples[:-1]
                    test += [samples[-1]]
                yield train, test

    def ui(
            self,
            train_cnt,
            iterations):
        """ """
        for sname1 in self.snames:
            for _ in range(iterations):
                train = []
                test = []
                for gname, samples in self.sgestures[sname1].items():
                    train += RS(samples, train_cnt)
                for sname2 in self.snames:
                    if sname1 == sname2:
                        continue
                    for gname, samples in self.sgestures[sname2].items():
                        test += RS(samples, 1)
                yield train, test

    def run_filter(self):
        for sample in self.samples:
            r = utils.get_cma_r(30.0, 3.0)
            sample.trajectory = utils.rcma(sample.trajectory, 1, r)


class Sample(object):
    """ """

    def __init__(
            self,
            sname,
            gname,
            ename,
            pid,
            trajectory,
            time_s):
        """ """
        self.sname = sname
        self.gname = gname
        self.ename = ename
        self.pid = pid
        self.trajectory = trajectory
        self.time_s = time_s

    def __iter__(self):
        """ """
        return iter(self.trajectory)

    @property
    def fps(self):
        """ """
        delta_s = self.time_s[-1] - self.time_s[0]
        frame_cnt = float(len(self.trajectory))
        return (frame_cnt - 1.0) / delta_s

    @classmethod
    def Load(
            cls,
            sname,
            gname,
            ename,
            epath,
            pid):
        """ """
        trajectory = []
        times_s = []

        with open(epath, "rb") as fin:

            str_cnt, = struct.unpack(
                "<B",
                fin.read(1))

            _gname, = struct.unpack(
                "{}s".format(str_cnt),
                fin.read(str_cnt))

            speed, pt_cnt, component_cnt = struct.unpack(
                "<III",
                fin.read(4 * 3))

            #
            # Load trajectory
            #
            bad = []
            for pt_no in range(pt_cnt):
                pt = []
                for component_no in range(component_cnt):
                    val, = struct.unpack(
                        "<f",
                        fin.read(4))
                    pt += [val]

                if np.count_nonzero(pt) == 0:
                    bad += [pt_no]
                    continue

                if len(trajectory) and np.array_equal(trajectory[-1], pt):
                    bad += [pt_no]
                    continue

                trajectory += [np.array(pt)]

            #
            # Load time stamps
            #
            timestamp_cnt, = struct.unpack(
                "<I",
                fin.read(4))
            assert timestamp_cnt == pt_cnt

            for pt_no in range(timestamp_cnt):

                val, = struct.unpack(
                    "<f",
                    fin.read(4))
                if pt_no in bad:
                    continue
                times_s += [val]

            assert len(times_s) == len(trajectory)

        ret = cls(
            sname,
            gname,
            ename,
            pid,
            np.array(trajectory),
            np.array(times_s))

        ret.fname = epath

        return ret
