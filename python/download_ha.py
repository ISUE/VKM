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
import zipfile
import urllib.request


def download_and_unzip(url, path):
    print(f"Downloading {url}...")
    filename = url.split("/")[-1]
    filepath = os.path.join(path, filename)
    urllib.request.urlretrieve(url, filepath)
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(path)


def dowload_ha(path):
    """
    Downloads the High Activity (HA) Kinect, Mouse, Vive Position and Vive Quaternion dataset
    into the specified directory.
    Dataset is available at: https://www.eecs.ucf.edu/isuelab/research/vkm/
    """

    target_dir = os.path.join(path)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    url = "https://www.eecs.ucf.edu/isuelab/research/vkm/VKM_datasets_kinect_mouse_vive.zip"
    download_and_unzip(url, target_dir)


if __name__ == "__main__":
    dowload_ha("../")
