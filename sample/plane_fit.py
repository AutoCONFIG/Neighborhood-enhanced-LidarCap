# -*- coding: utf-8 -*-
# @Author  : jingyi

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import open3d as o3d
import numpy as np
import random, h5py
import configargparse
from util.smpl import SMPL
from tqdm import tqdm
import math


def fit_plane(planes):
    _plane = np.zeros((0, 4))
    for i in tqdm(range(planes.shape[0])):
        plane = planes[i] #(512, 3)
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(plane)
        plane_model, _ = points.segment_plane(distance_threshold=0.01,
                                              ransac_n=3,
                                              num_iterations=1000)
        _plane = np.concatenate((_plane, plane_model[np.newaxis, :]))

    return _plane


