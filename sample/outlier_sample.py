#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Zhang Jingyi
Reference: https://blog.csdn.net/McQueen_LT/article/details/118298329
'''

import open3d as o3d
import numpy as np
FLT_EPSILON = 1e-1
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import open3d as o3d
import numpy as np
from tqdm import tqdm
import configargparse
#from util.smpl import SMPL
import random
'''
Inputs:
    points_array: all points array
    point_array: single point
    neighbor_idx: kdtree nearest indice
    u,v: getCoordinateSystemOnPlane
    angle_threshold: rad
Outputs:
    True or False
Reference: https://pointclouds.org/documentation/boundary_8hpp_source.html
'''


def isBoundaryPoint(points_array, point_array, neighbor, u, v, angle_threshold):
    points_array = points_array.squeeze()
    neighbor_idx = np.array(neighbor)
    if neighbor_idx.size <= 3: return False
    max_dif = 0
    cp = 0
    angle_array = np.zeros(neighbor_idx.size)
    for i in range(neighbor_idx.size):
        delta = points_array[neighbor_idx[i]] - point_array # 1 * 3 XYZ
        if all(delta.any() == np.zeros(3)): continue
        angle = np.arctan2(np.dot(v, delta.T), np.dot(u, delta.T)) # (rad)the angles are fine between -PI and PI too
        angle_array[cp] = angle
        cp = cp + 1
    if cp == 0: return False
    angle_array = np.resize(angle_array, cp)
    angle_array = np.sort(angle_array) # 升序
    # Compute the maximal angle difference between two consecutive angles
    for i in range(angle_array.size - 1):
        dif = angle_array[i+1] - angle_array[i]
        if max_dif < dif: max_dif = dif
    # Get the angle difference between the last and the first
    dif = 2 * np.pi - angle_array[angle_array.size - 1] + angle_array[0]
    if max_dif < dif: max_dif = dif

    return (max_dif > angle_threshold)

'''
Inputs: tuple
Outputs: array( np.array([[]]) )
i.e: np.array([[1,2,3]])
# L109
# https://pointclouds.org/documentation/cuda_2common_2include_2pcl_2cuda_2common_2eigen_8h_source.html
'''
def unitOrthogonal(tuple):
    def isMuchSmallerThan(x, y):
        global FLT_EPSILON
        prec_sqr = FLT_EPSILON * FLT_EPSILON
        if x*x <= prec_sqr*y*y:
            return True
        else:
            return False
    if (not isMuchSmallerThan(tuple[0], tuple[1])) or (not isMuchSmallerThan(tuple[1], tuple[2])):
        invum = 1.0 / np.sqrt(tuple[0]**2 + tuple[1]**2)
        perp_x = -tuple[1] * invum
        perp_y = tuple[0] * invum
        perp_z = 0.0
    else:
        invum = 1.0 / np.sqrt(tuple[2]**2 + tuple[1]**2)
        perp_x = 0.0
        perp_y = -tuple[2] * invum
        perp_z = tuple[1] * invum
    perp_array = np.array([[perp_x, perp_y, perp_z]])
    return perp_array

'''
Inputs: point_normal(pcl.PointCloud_Normal)注意这里的normal已经是单位向量了而且是个tuple
pcl::Normal::Normal	(	
float 	n_x,
float 	n_y,
float 	n_z,
float 	_curvature
)		
Outputs: u and v
'''
def getCoordinateSystemOnPlane(point_normal):
    normal_tuple = point_normal
    # v = p_coeff_v.unitOrthogonal();
	# u = p_coeff_v.cross3(v);
    # v是normal的正交向量
    v = unitOrthogonal(normal_tuple)
    u = np.cross(point_normal[:3], v)
    return u, v

'''
Inputs: pointCloud(pcl.pointCloud)
Outputs: normals(pcl.PointCloud_Normal)
Reference by demo NormalEstimation.py
'''
def compute_normals(pointCloud):
    # downpcd = pointCloud.voxel_down_sample(voxel_size=0.05)
    pointCloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pointCloud
'''
Inputs: 3D-pointcloud(points array) (n*3)XYZ
Outputs: 3D-pointcloud(points array) (n*3)XYZ
'''


def outline_onesample(points):
    _points = points  # (512,3)
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(_points)
    points = compute_normals(points)
    # Build a kdtree
    pc_kdtree = o3d.geometry.KDTreeFlann(points)

    boundary_label = np.zeros((0, 1))
    for i in range(len(points.points)):
        # Find nearestKSearch points
        K = 40
        # searchPoint = o3d.geometry.PointCloud()
        # searchPoint.points=o3d.utility.Vector3dVector(np.array([points.points[i]]))
        [neighbor_dist, neighbor_idx, _] = pc_kdtree.search_knn_vector_3d(
            points.points[i], K)
        # pc[neighbor_idx[0][i]]
        # getCoordinateSystemOnPlane, Obtain a coordinate system on the least-squares plane
        u, v = getCoordinateSystemOnPlane(points.normals[i])
        # isBoundaryPoint
        label = isBoundaryPoint(np.array([points.points]), np.array([points.points[i]]),
                                neighbor_idx, u, v, angle_threshold=1.5)
        boundary_label = np.vstack([boundary_label, label])
    return boundary_label


def boundary_detection(pointclouds):
    
    boundary = np.zeros((0,pointclouds.shape[1],1))
    # random_pc = np.zeros((0, pointclouds.shape[1],3))
    for index in tqdm(range(pointclouds.shape[0])):
        boundary_label = outline_onesample(pointclouds[index])
        boundary = np.vstack([boundary, boundary_label[np.newaxis,]])
        # _random_point = random_point(pointclouds[index], boundary_label)
        # random_pc = np.vstack([random_pc, _random_point[np.newaxis,]])
    return boundary

import h5py


def array_to_pointcloud(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def conv_hull(points: np.ndarray):
    pcl = array_to_pointcloud(points)
    hull, lst = pcl.compute_convex_hull()
    return lst


def in_convex_polyhedron(points_set: np.ndarray, test_points: np.ndarray):
    """
    检测点是否在凸包内
    :param points_set: 凸包，需要对分区的点进行凸包生成 具体见conv_hull函数
    :param test_points: 需要检测的点 可以是多个点
    :return: bool类型
    """
    assert type(points_set) == np.ndarray
    assert type(points_set) == np.ndarray
    bol = np.zeros((test_points.shape[0], 1), dtype=np.bool)
    ori_set = points_set
    ori_edge_index = conv_hull(ori_set)
    ori_edge_index = np.sort(np.unique(ori_edge_index))
    for i in range(test_points.shape[0]):
        new_set = np.concatenate((points_set, test_points[i, np.newaxis]), axis=0)
        new_edge_index = conv_hull(new_set)
        new_edge_index = np.sort(np.unique(new_edge_index))
        bol[i] = (new_edge_index.tolist() == ori_edge_index.tolist())
    return bol


def random_point(points, boundary_label_):
    inside_points = points[~boundary_label_.squeeze().astype(bool)]
    boundary_points = points[boundary_label_.squeeze().astype(bool)]
    random_noise_1 = np.random.rand(inside_points.shape[0],
                                    inside_points.shape[1]) * 0.15
    random_noise_2 = np.random.rand(inside_points.shape[0],
                                    inside_points.shape[1]) * 0.15
    random_noise = random_noise_2 - random_noise_1
    random_noise[:, -1:] = random_noise[:, -1:] * 2
    random_inside = inside_points + random_noise
    try:
        bool_ = in_convex_polyhedron(boundary_points, random_inside)
        final_inside = inside_points * ~bool_ + random_inside * bool_
        random = np.concatenate((final_inside, boundary_points))
    except:
        random = points
    return random
    
    
def deal_file(index):
    
    print(f'{index} start!')
    file_path = os.path.join(args.file_path, str(index) + '_bg.hdf5')
    f = h5py.File(file_path, 'r')
    pc = f['point_clouds']
    out = os.path.join(args.out_path, str(index) + '.hdf5')
    boundary = boundary_detection(pc)
    
    ff = h5py.File(out, 'w')
    ff['boundary_label'] = boundary # B, N, 1
    # ff['sample_pc'] = f['sample_pc'][:]
    # ff['random_pc'] = random_pc # B, N, 3
    ff['background_m'] = f['backgrounds'][:]
    ff['full_joints'] = f['full_joints'][:]
    ff['point_clouds'] = f['point_clouds'][:]
    ff['points_num'] = f['points_num'][:]
    ff['pose'] = f['pose'][:]
    ff['rotmats'] = f['rotmats'][:]
    ff['shape'] = f['shape'][:]
    ff['trans'] = f['trans'][:]
    # f.close()
    ff.close()
    # print(f'{index} down!')


def get_range(intial,P,boundary,error_index):
    for i in range(len(error_index)):
         if intial<P<boundary[error_index[i+1]]:
             range_y = [intial-P,boundary[error_index[i+1]]-P]
             return range_y
         elif boundary[error_index[i+1]]< P<intial:
             range_y = [boundary[error_index[i+1]]-P,intial-P]
             return range_y
         else:
             return [0,0]


def add_dis_xy(P_x_, P_y_,boundary_x_,boundary_y_):
    error_x_index = np.argsort(np.abs(boundary_x_ - P_x_))
    intial = boundary_y_[error_x_index[0]]
    range_y = get_range(intial, P_y_, boundary_y_, error_x_index)
    random_dis = random.uniform(range_y[0], range_y[1])
    return P_y_+random_dis


def add_dis_z(P_x_, P_y_,P_z_,boundary_x_,boundary_y_,boundary_z_):
    error_x_index = np.argsort(((boundary_x_ - P_x_)**2+(boundary_y_ - P_y_)**2)**0.5)
    intial = boundary_z_[error_x_index[0]]
    range_y = get_range(intial, P_z_, boundary_z_, error_x_index)
    random_dis = random.uniform(range_y[0], range_y[1])
    return P_z_+random_dis


def test_boundary():
    points = np.loadtxt('sample/1409_522.txt')
    bound_inds = outline_onesample(points)

    random = random_point(points, bound_inds)
    boundary = points[bound_inds.astype(bool).squeeze()]

    np.savetxt('sample/b_boundary.txt', boundary)
    # inside_pc = points[~(bound_inds.astype(bool).squeeze())]
    #
    # boundary_x, boundary_y, boundary_z, = boundary[:, 0], boundary[:, 1], boundary[:, 2]
    # # P  = inside_pc[0]   #(,3) x y z
    #
    # random_noise = np.zeros_like(inside_pc)
    # for k in range(inside_pc.shape[0]):
    #     # print (k)
    #     point = inside_pc[k]
    #     P_x, P_y, P_z = point[0], point[1], point[2]
    #     P_y = add_dis_xy(P_x, P_y, boundary_x, boundary_y)
    #     P_x = add_dis_xy(P_y, P_x, boundary_y, boundary_x)
    #     P_z = add_dis_z(P_x, P_y, P_z, boundary_x, boundary_y, boundary_z)
    #     random_noise[k] = np.array([P_x, P_y, P_z])

    np.savetxt('sample/sample_noise.txt', random)
