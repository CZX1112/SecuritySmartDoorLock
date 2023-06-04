# -*- coding: utf-8 -*-
from __future__ import print_function  # do not delete this line if you want to save your log file.
import moxing as mox
import zipfile
import os
import subprocess
from naie.context import Context


def load_data():
    
    from naie.datasets import get_data_reference
    data_reference = get_data_reference(dataset="faces_train", dataset_entity="faces_train")
    file_paths = data_reference.get_files_paths()
    mox.file.copy(data_reference.get_files_paths()[0], "/cache/faces_train.zip")
    rf = zipfile.ZipFile("/cache/faces_train.zip")
    rf.extractall("/cache/")
    rf.close()
    print(file_paths)


def main():
    config = './configs/resnet/resnet18_b32x8_imagenet.py'
    gpus = 8
    workdir = './ckpt'

    cmd = " {} {} --work-dir {} ".format(config, gpus, workdir)
    print(cmd)
    subprocess.call("cd mmclassification-master && pip install -e . ", shell = True)
    subprocess.call("cd mmclassification-master && sh ./tools/dist_train.sh " + cmd, shell = True)

if __name__ == "__main__":
    load_data()
    main()
    mox.file.copy('/cache/user-job-dir/codes/mmclassification-master/ckpt/latest.pth', os.path.join(Context.get_model_path(), 'latest.pth'))
    os.path.abspath('.')
    print("path:", os.path.abspath('.'))
    print("dir:", os.listdir(os.path.abspath('.')))