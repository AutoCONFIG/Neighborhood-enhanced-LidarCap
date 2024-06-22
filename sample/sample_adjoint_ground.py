# -*- coding: utf-8 -*-
# @Author  : jingyi
import configargparse,h5py,os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import open3d as o3d


def background_near(points, trans, background, num_points=512):
    background_points=np.zeros((0,512,3))
    show = []
    for index in tqdm(range(trans.shape[0])):
        location = trans[index]
        x_min = location[0]-2
        x_max = location[0]+2
        y_min = location[1]-2
        y_max = location[1]+2
        back_x_index = np.logical_and(background[:,0]>x_min,background[:,0]<x_max)
        back_y_index = np.logical_and(background[:,1]>y_min,background[:,1]<y_max)
        back_index = np.logical_and(back_y_index, back_x_index)
        back = background[back_index]
        origin_num_points = back_index.sum()

        if origin_num_points < num_points:
            num_whole_repeat = num_points // origin_num_points
            res = back.repeat(num_whole_repeat, axis=0)
            num_remain = num_points % origin_num_points
            res = np.vstack((res, res[:num_remain]))
        if origin_num_points >= num_points:
            res = back[np.random.choice(origin_num_points, num_points)]
        background_points = np.concatenate((background_points, res[np.newaxis,:]))

    return background_points


def background_auto(points, trans, background_path, init_frame, num_points=512):

    background_points=np.zeros((0,512,3))
    show = []
    for index in tqdm(range(trans.shape[0])):

        back_frame = int(init_frame)+int(index)
        background_file = os.path.join(background_path, str("%06d" % back_frame)+'.pcd')
        tmp = o3d.io.read_point_cloud(background_file)
        background = np.array(tmp.points)

        location = trans[index]
        x_min = location[0]-2
        x_max = location[0]+2
        y_min = location[1]-2
        y_max = location[1]+2
        back_x_index = np.logical_and(background[:,0]>x_min,background[:,0]<x_max)
        back_y_index = np.logical_and(background[:,1]>y_min,background[:,1]<y_max)
        back_index = np.logical_and(back_y_index, back_x_index)
        back = background[back_index]
        for point in points[index]:
            back = np.delete(back, np.where(back==point), axis=0)

        origin_num_points = back.shape[0]

        if origin_num_points < num_points:
            num_whole_repeat = num_points // origin_num_points
            res = back.repeat(num_whole_repeat, axis=0)
            num_remain = num_points % origin_num_points
            res = np.vstack((res, res[:num_remain]))
        if origin_num_points >= num_points:
            res = back[np.random.choice(origin_num_points, num_points)]
        background_points = np.concatenate((background_points, res[np.newaxis,:]))

    return background_points


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument("--file_path", '-F', type=str,
                        default='your_data_path')

    parser.add_argument("--out_path", '-O', type=str,
                        default='your_save_path')

    parser.add_argument("--vis", action='store_true', default=False)

    parser.add_argument("--manual", action='store_true', default=False)

    parser.add_argument("--vis_file_index", default=None, type=str)

    args = parser.parse_args()
    
    if args.manual:
        # train_set = [
        #     51801, 51802, 51803, 51808, 51807, 61102, 61103, 61105, 61108, 61109, 61201,
        #     61202, 61203, 61206, 61207, 61210, 61212, 61213, 61301, 61302, 61304, 61305,
        #     61306, 61307, 61704, 61705, 61706, 61707, 61708, 61709, 61711, 61714, 61801,
        #     61802, 61803, 61804, 61805, 61808, 51809, 51804, 51812, 51810
        # ]
        # test_set = [61107, 61101, 61204, 61209, 61303, 61702, 61713, 61807, 61806, 61106
        #             ]
        # all_set = train_set + test_set
        all_set = [51804]

        background_path = 'your_data_path'

        for index in tqdm(all_set):
            file_path = os.path.join(args.file_path, str(index) + '.hdf5')
            f = h5py.File(file_path, 'r')
            out = os.path.join(args.out_path, str(index) + '.hdf5')

            background_file = os.path.join(background_path, str(index) + '.txt')
            background = np.loadtxt(background_file)
            background_points=background_near(points=f['point_clouds'],trans=f['trans'],background=background[:,:3])
            # sample_pc, boundary_label = simulatorLiDAR(f, out_root=out, shorter_dist=0,
            #                                            move_z=0, rot=np.eye(3), threads=1)
            ff = h5py.File(out, 'w')
            ff['boundary_label'] = f['boundary_label'][:]
            ff['background_points'] = background_points
            ff['full_joints'] = f['full_joints'][:]
            ff['point_clouds'] = f['point_clouds'][:]
            ff['points_num'] = f['points_num'][:]
            ff['pose'] = f['pose'][:]
            ff['rotmats'] = f['rotmats'][:]
            ff['shape'] = f['shape'][:]
            ff['trans'] = f['trans'][:]
            f.close()
            ff.close()
            print(f'{index} down!')
    else:
        train_set = [
            51801, 51802, 51803, 51808, 51807, 61102, 61103, 61105, 61108, 61109, 61201,
            61202, 61203, 61206, 61207, 61210, 61212, 61213, 61301, 61302, 61304, 61305,
            61306, 61307, 61704, 61705, 61706, 61707, 61708, 61709, 61711, 61714, 61801,
            61802, 61803, 61804, 61805, 61808, 51809, 51804, 51812, 51810
        ]
        test_set = [61107, 61101, 61204, 61209, 61303, 61702, 61713, 61807, 61806, 61106
                    ]
        all_set = train_set + test_set

        init_frame = {'51801':43, '51802':21, '51803':40, '51808':293, '51807':337,
                      '61102':43, '61103':35, '61105':34, '61108':40, '61109':213,
                      '61201':106, '61202':19, '61203':18, '61206':23, '61207':271,
                      '61210':196, '61212':68, '61213':80, '61301':58, '61302':1838,
                      '61304':66, '61305':70, '61306':41, '61307':120, '61704':47,
                      '61705':22, '61706':51, '61707':49, '61708':29, '61709':45,
                      '61711':18, '61714':60, '61801':37, '61802':32, '61803':26,
                      '61804':27, '61805':28, '61808':37, '51809':453, '51804':27,
                      '51812':44, '51810':309, '61107':21, '61101':33, '61204':26,
                      '61209':395, '61303':167, '61702':98, '61713':21, '61807':23,
                      '61806':33, '61106':35}

        for index in tqdm(all_set):
            background_path = os.path.join('your_data_path', str(index))
            file_path = os.path.join(args.file_path, str(index) + '.hdf5')
            f = h5py.File(file_path, 'r')
            out = os.path.join(args.out_path, str(index) + '.hdf5')

            background_points=background_auto(points=f['point_clouds'], trans=f['trans'], background_path = background_path, init_frame=init_frame[str(index)])
            # sample_pc, boundary_label = simulatorLiDAR(f, out_root=out, shorter_dist=0,
            #                                            move_z=0, rot=np.eye(3), threads=1)
            ff = h5py.File(out, 'w')
            ff['boundary_label'] = f['boundary_label'][:]
            ff['background_points'] = background_points
            ff['full_joints'] = f['full_joints'][:]
            ff['point_clouds'] = f['point_clouds'][:]
            ff['points_num'] = f['points_num'][:]
            ff['pose'] = f['pose'][:]
            ff['rotmats'] = f['rotmats'][:]
            ff['shape'] = f['shape'][:]
            ff['trans'] = f['trans'][:]
            f.close()
            ff.close()
            print(f'{index} down!')