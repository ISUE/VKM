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


from enum import Enum


class DeviceType(Enum):
    KINECT = 0
    VIVE = 1
    VIVE_POSITION = 2
    VIVE_QUATERNION = 3
    JK2017_KINECT = 4
    JK2017_LEAP_MOTION = 5
    GDS = 6
    WII = 7
    MOUSE = 8
    SHREC19 = 9
    CHALEARN11 = 10
    CHALEARN14 = 11


class gpsr_coefficients_t:
    def __init__(self,
                 device,
                 pid,
                 intercept,
                 angle_co,
                 density_co,
                 density_angle_co,
                 best_ks):
        """ """
        self.device = device
        self.pid = pid
        self.intercept = intercept
        self.angle_co = angle_co
        self.density_co = density_co
        self.density_angle_co = density_angle_co
        self.best_ks = best_ks


gpsr_coefficients = gpsr_coefficients_t(
    0, -1, 4.471682, -0.686062, 0.464960, 0.384859, 0.146102)


def get_gpsr_coefficients() -> gpsr_coefficients_t:
    return gpsr_coefficients


def get_inflation(did: DeviceType):
    if did == DeviceType.KINECT:
        return 1.4
    if did == DeviceType.VIVE_POSITION:
        return 1.52
    if did == DeviceType.VIVE_QUATERNION:
        return 1.68
    if did == DeviceType.MOUSE:
        return 1.2


def get_device_resample_count(device_type: DeviceType):
    if device_type == DeviceType.KINECT:
        return 20
    if device_type == DeviceType.VIVE_POSITION:
        return 32
    if device_type == DeviceType.VIVE_QUATERNION:
        return 16
    if device_type == DeviceType.MOUSE:
        return 96


def get_device_dataset_path(device_type: DeviceType):
    if device_type == DeviceType.KINECT:
        return "../datasets/kinect/training"
    if device_type == DeviceType.GDS:
        return "../datasets/gds/training"
    # FIXME Vive pos?
    if device_type == DeviceType.VIVE:
        return "../datasets/vive/training",
    if device_type == DeviceType.MOUSE:
        return "../datasets/mouse/training"
