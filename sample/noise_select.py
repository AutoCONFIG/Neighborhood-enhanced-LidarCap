from ast import parse
from plyfile import PlyData, PlyElement
from typing import List
import argparse
import numpy as np
import json
import os
import re
import sys
import h5py
import torch
import pickle as pkl

from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util.multiprocess import multi_func
from util.smpl import SMPL
from scipy.spatial.transform import Rotation as R


ROOT_PATH = 'your_dataset_path'
MAX_PROCESS_COUNT = 16

# img_filenames = []

import open3d as o3d

transet = [5, 6, 8, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42]
testset = [7, 24, 29, 41]
# os.makedirs(extras_path, exist_ok=True)
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
parser = argparse.ArgumentParser()

# parser.add_argument('--seqlen', type=int, default=16)
parser.add_argument('--npoints', type=int, default=512)
parser.add_argument('--ids', nargs='+', default=testset)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--noise', action='store_true', default=False)
# parser.set_defaults(func=dump)

args = parser.parse_args()

def read_point_cloud(filename):
    return np.asarray(o3d.io.read_point_cloud(filename).points)


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def save_ply(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2])
              for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=False).write(filename)


def get_index(filename):
    basename = os.path.basename(filename)
    return int(os.path.splitext(basename)[0])


def get_sorted_filenames_by_index(dirname, isabs=True):
    if not os.path.exists(dirname):
        return []
    filenames = os.listdir(dirname)
    filenames = sorted(os.listdir(dirname), key=lambda x: get_index(x))
    if isabs:
        filenames = [os.path.join(dirname, filename) for filename in filenames]
    return filenames


def get_paths_by_suffix(dirname, suffix):
    filenames = list(filter(lambda x: x.endswith(suffix), os.listdir(dirname)))
    assert len(filenames) > 0
    return [os.path.join(dirname, filename) for filename in filenames]


def parse_json(json_filename):
    with open(json_filename) as f:
        content = json.load(f)
        beta = np.array(content['beta'], dtype=np.float32)
        pose = np.array(content['pose'], dtype=np.float32)
        trans = np.array(content['trans'], dtype=np.float32)
    return beta, pose, trans


def parse_pkl(pkl_filename):
    mocap2lidar_matrix = np.array([
        [-1,0,0],
        [0,0,1],
        [0,1,0]])

    behave2lidarcap = np.array([
        [ 0.34114292, -0.81632519,  0.46608443],
        [-0.93870972, -0.26976026,  0.21460074],
        [-0.04945297, -0.5107275, -0.85831922]])
    with open(pkl_filename, 'rb') as f:
        content = pkl.load(f)
        beta = np.array(content['betas'], dtype=np.float32) # (10, )
        pose = np.array(content['pose'][:72], dtype=np.float32) # (72, )
        trans = np.array(content['trans'], dtype=np.float32)# (3, )
    pose[:3]=(R.from_matrix(mocap2lidar_matrix) * R.from_rotvec(pose[:3])).as_rotvec()
    pose[:3]=(R.from_matrix(behave2lidarcap) * R.from_rotvec(pose[:3])).as_rotvec()
    return beta, pose, trans


def fix_points_num(points: np.array, num_points: int):
    points = points[~np.isnan(points).any(axis=-1)]

    origin_num_points = points.shape[0]
    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    if origin_num_points >= num_points:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res


def compute_dist(a, b):
    pcda = o3d.geometry.PointCloud()
    pcda.points = o3d.utility.Vector3dVector(a)
    pcdb = o3d.geometry.PointCloud()
    pcdb.points = o3d.utility.Vector3dVector(b)
    return np.asarray(pcda.compute_point_cloud_distance(pcdb))


def foo(id, npoints):
    id = str(id)

    smpl = SMPL().cuda()

    # cur_img_filenames = get_sorted_filenames_by_index(
    #     os.path.join(ROOT_PATH, 'images', id))

    pose_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'pose', id))
    json_filenames = list(filter(lambda x: x.endswith('json'), pose_filenames))
    #ply_filenames = list(filter(lambda x: x.endswith('ply'), pose_filenames))

    '''cur_betas, cur_poses, cur_trans = multiprocess.multi_func(
        parse_json, MAX_PROCESS_COUNT, len(json_filenames), 'Load json files',
        True, json_filenames)
    # cur_vertices = multiprocess.multi_func(
    #     read_ply, MAX_PROCESS_COUNT, len(ply_filenames), 'Load vertices files',
    #     True, ply_filenames)

    depth_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'depth', id))
    cur_depths = depth_filenames'''

    static_bg_path = os.path.join('your_dataset_path',id,'bg.pcd')
    static_bg = read_point_cloud(static_bg_path)

    segment_filenames = get_sorted_filenames_by_index(
        os.path.join(ROOT_PATH, 'labels', '3d', 'segment', id))
    cur_point_clouds = multi_func(
        read_point_cloud, MAX_PROCESS_COUNT, len(segment_filenames),
        'Load segment files', True, segment_filenames)

    bg_filenames = [
        os.path.join(f'your_dataset_path/{id}', os.path.basename(e).replace('.ply', '.pcd')) for
        e in segment_filenames]

    bgs = multi_func(
        read_point_cloud, MAX_PROCESS_COUNT, len(bg_filenames),
        'Load bg files', True, bg_filenames)

    whole_noise = []
    whole_bg_points_nums = []
    whole_static_bg = []

    static_bg_angle = np.zeros((static_bg.shape[0], static_bg.shape[1] + 3))
    static_bg_angle[:, :3] = static_bg
    static_bg_angle[:, 3] = np.degrees(np.arctan2(static_bg[:, 0], static_bg[:, 1]))
    dis = np.linalg.norm(static_bg[:, :2], axis=1, keepdims=True)
    static_bg_angle[:, 4] = np.degrees(np.arctan2(static_bg[:, 2], dis.squeeze()))
    static_bg_angle[:, 5] = np.linalg.norm(static_bg, axis=1, keepdims=True).squeeze()

    for human_points, bg in tqdm(list(zip(cur_point_clouds, bgs)), desc='Gen Background', ncols=60):
        # human_points: (n, 3)
        # bd: (N, 3)
        human_points_angle = np.zeros((human_points.shape[0], human_points.shape[1]+3))
        human_points_angle[:,:3] = human_points
        human_points_angle[:,3] = np.degrees(np.arctan2(human_points[:,0], human_points[:,1]))
        dis = np.linalg.norm(human_points[:,:2], axis=1,keepdims=True)
        human_points_angle[:,4] = np.degrees(np.arctan2(human_points[:,2],dis.squeeze()))
        human_points_angle[:,5] = np.linalg.norm(human_points, axis=1,keepdims=True).squeeze()
        # human_points_angle (x-y, z-xy, r)

        bg_angle = np.zeros((bg.shape[0], bg.shape[1]+3))
        bg_angle[:,:3] = bg
        bg_angle[:,3] = np.degrees(np.arctan2(bg[:,0], bg[:,1]))
        dis = np.linalg.norm(bg[:,:2], axis=1,keepdims=True)
        bg_angle[:,4] = np.degrees(np.arctan2(bg[:,2],dis.squeeze()))
        bg_angle[:,5] = np.linalg.norm(bg, axis=1,keepdims=True).squeeze()

        angle_1_min = min(human_points_angle[:,3])-1
        angle_1_max = max(human_points_angle[:,3])+1

        angle_2_min = min(human_points_angle[:,4])-1
        angle_2_max = max(human_points_angle[:,4])+1

        index_1 = np.logical_and(bg_angle[:, 3] > angle_1_min,
                                 bg_angle[:, 3] < angle_1_max)
        index_2 = np.logical_and(bg_angle[:, 4] > angle_2_min,
                                 bg_angle[:, 4] < angle_2_max)
        index_all = np.logical_and(index_2, index_1)

        if args.noise:

            bg_noise = np.zeros((index_all.sum(),bg_angle.shape[1]+2))
            bg_noise[:,:6]=bg_angle[index_all]

            for points in human_points_angle:
                a = (bg_noise[:,:3]==points[:3])
                index = np.logical_and(a[:,0],np.logical_and(a[:,1],a[:,2]))
                bg_noise[index,-2]=1

                points_1_min = points[3]-1
                points_1_max = points[3]+1
                points_2_min = points[4]-1
                points_2_max = points[4]+1

                _index_1 = np.logical_and(bg_noise[:, 3] > points_1_min,
                                         bg_noise[:, 3] < points_1_max)
                _index_2 = np.logical_and(bg_noise[:, 4] > points_2_min,
                                         bg_noise[:, 4] < points_2_max)

                _index_all = np.logical_and(np.logical_and(_index_2, _index_1),
                                            bg_noise[:,5]>points[5])

                bg_noise[_index_all, -1] = 2

            noice = bg_noise[bg_noise[:,-1]+bg_noise[:,-2] == 2][...,:3]

        else:
            bg_noise = bg.copy()[index_all]
            for point in human_points:
                a = (bg_noise==point)
                index_human = np.logical_and(a[:,0],np.logical_and(a[:,1],a[:,2]))
                bg_noise=bg_noise[~index_human]


            static_index_1 = np.logical_and(static_bg_angle[:, 3] > angle_1_min,
                                            static_bg_angle[:, 3] < angle_1_max)
            static_index_2 = np.logical_and(static_bg_angle[:, 4] > angle_2_min,
                                            static_bg_angle[:, 4] < angle_2_max)
            static_index_all = np.logical_and(static_index_2, static_index_1)

            static_ = static_bg[static_index_all]

        whole_noise.append(fix_points_num(bg_noise, npoints))
        whole_static_bg.append(fix_points_num(static_, npoints))

    return np.stack(whole_noise), np.stack(whole_static_bg)


def get_sorted_ids(s):
    if re.match('^([1-9]\d*)-([1-9]\d*)$', s):
        start_index, end_index = s.split('-')
        indexes = list(range(int(start_index), int(end_index) + 1))
    elif re.match('^(([1-9]\d*),)*([1-9]\d*)$', s):
        indexes = [int(x) for x in s.split(',')]
    return sorted(indexes)


def dump(ids, npoints, name):
    #ids = [1, 2, 3, 410, 4, 50301, 50302, 50304, 50305, 50306, 50307, 50308]
    whole_noise_twice = np.zeros((0, npoints, 3))
    whole_static_bg = np.zeros((0, npoints, 3))

    for id in ids:
        print('start process', id)

        noise_twice, static_bg = foo(id, npoints)
        """
        import OVis
        human_points = point_clouds.copy()
        human_points[np.logical_not(body_mask)] = 0
        OVis.ovis.magic((rotmats, trans), (point_clouds + 0.5), (human_points - 0.5))"""

        whole_noise_twice = np.concatenate((whole_noise_twice, noise_twice))
        whole_static_bg = np.concatenate((whole_static_bg, static_bg))

    whole_filename = name + '.hdf5'
    with h5py.File(os.path.join('dataset/back_mqh', whole_filename), 'a') as f:
        f['whole_shadow'] = whole_noise_twice
        f['whole_static_bg'] = whole_static_bg
    f.close()
    print('Success add keys:', os.path.join('dataset/back_mqh', whole_filename))


def fix_to_seq16(dataset_path, dataset_ids):
    d = h5py.File(dataset_path, 'r')
    datasets_length = [len(get_sorted_filenames_by_index(os.path.join(ROOT_PATH, 'labels', '3d', 'pose', str(idx)))) for idx in dataset_ids]
    with h5py.File(dataset_path + '_fixed', 'w') as f:
        for k, v in d.items():
            seq = v[:]
            chunks = np.split(seq, np.cumsum(datasets_length[:-1]))
            chunks = [np.concatenate((e, np.repeat(e[-2:-1], (16 - len(e) % 16) % 16, axis=0)), axis=0) for e in chunks]
            seq = np.concatenate(chunks, axis=0)
            f.create_dataset(k, data=seq)
            print(v.shape, '->', seq.shape)
    print()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    dump(args.ids, args.npoints, 'lidarcap_test')


