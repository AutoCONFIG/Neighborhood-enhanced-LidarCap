import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import open3d as o3d
import numpy as np
from glob import glob
import configargparse
from multiprocessing import Pool
import functools
import time, torch
from util.smpl import SMPL
from sample.outlier_sample import outline_onesample
smpl=SMPL()

def hidden_point_removal(pcd, camera = [0, 0, 0]):
    # diameter = np.linalg.norm(
    # np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    # print("Define parameters used for hidden_point_removal")
    
    # camera = [view_point[0], view_point[0], diameter]
    # camera = view_point
    dist = np.linalg.norm(pcd.get_center())
    # radius = diameter * 100
    radius = dist * 1000

    # print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    # print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    return pcd


def select_points_on_the_scan_line(points, view_point=None, scans=64, line_num=1024, fov_up=16.2, fov_down=-16.2, precision=1.1):
    
    fov_up = np.deg2rad(fov_up)
    fov_down = np.deg2rad(fov_down)
    fov = abs(fov_down) + abs(fov_up)

    ratio = fov/(scans - 1)   # 64bins 的竖直分辨率
    hoz_ratio = 2 * np.pi / (line_num - 1)    # 64bins 的水平分辨率
    # precision * np.random.randn() 

    # print(points.shape[0])
    
    if view_point is not None:
        points -= view_point
    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    
    # pc_ds = []

    saved_box = { s:{} for s in np.arange(scans)}

    #### 筛选fov范围内的点
    for idx in range(0, points.shape[0]):
        rule1 =  pitch[idx] >= fov_down
        rule2 =  pitch[idx] <= fov_up
        rule3 = abs(pitch[idx] % ratio) < ratio * 0.4
        rule4 = abs(yaw[idx] % hoz_ratio) < hoz_ratio * 0.4
        if rule1 and rule2:
            scanid = np.rint((pitch[idx] + 1e-4) / ratio) + scans // 2
            pointid = np.rint((yaw[idx] + 1e-4) // hoz_ratio)

            if pointid > 0 and scan_x[idx] < 0:
                pointid += 1024 // 2
            elif pointid < 0 and scan_y[idx] < 0:
                pointid += 1024 // 2
            
            z = np.sin(scanid * ratio + fov_down)
            xy = abs(np.cos(scanid * ratio + fov_down))
            y = xy * np.sin(pointid * hoz_ratio)
            x = xy * np.cos(pointid * hoz_ratio)

            # 找到根指定激光射线夹角最小的点
            cos_delta_theta = np.dot(points[idx], np.array([x, y, z])) / depth[idx]
            delta_theta = np.arccos(abs(cos_delta_theta))
            if pointid in saved_box[scanid]:
                if delta_theta < saved_box[scanid][pointid]['delta_theta']:
                    saved_box[scanid][pointid].update({'points': points[idx], 'delta_theta': delta_theta})
            else:
                saved_box[scanid][pointid] = {'points': points[idx], 'delta_theta': delta_theta}

    save_points  =[]
    for key, value in saved_box.items():
        if len(value) > 0:
            for k, v in value.items():
                save_points.append(v['points']) 

    # pc_ds = np.array(pc_ds)
    save_points = np.array(save_points)


    #####
    # print(f'\r{save_points.shape}', end=' ', flush=True)
    pc=o3d.open3d.geometry.PointCloud()
    pc.points= o3d.open3d.utility.Vector3dVector(save_points)
    pc.paint_uniform_color([0.5, 0.5, 0.5])
    # pc.estimate_normals()

    return pc

def translate(points, y_dist=5, z_height=3.1):
    """
    params: 
        向human向Y轴拉近 @y_dist, 向上平移 @z_height
    returns: 
        points
    """
    points.translate(np.array([0, -y_dist, z_height]))
    return points

def sample_data(pc, shorter_dist, move_z, rot):

    # save data path
    # filename, _ = os.path.splitext(os.path.basename(file_path))
    # save_path = os.path.join(out_root, filename + '.pcd')
    # if os.path.exists(save_path):
    #     return
    # 
    # print(f'\rProcess {file_path}', end='\t', flush=True)

    time1 = time.time()

    # point_clouds = o3d.open3d.io.read_triangle_mesh(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # point_clouds.compute_vertex_normals()
    # point_clouds = point_clouds.sample_points_poisson_disk(100000)
    # time2 = time.time()

    # print(f'Read {(time2- time1):.3f} s.')

    # if len(point_clouds.triangles) > 0:
    #     point_clouds.compute_vertex_normals()
    #     point_clouds = point_clouds.sample_points_poisson_disk(100000)
    # else:
    #     point_clouds = o3d.io.read_point_cloud(file_path)
    #     

    # point_clouds
    # view_point = point_clouds.get_center()
    # view_point[0] += 0
    # view_point[1] += -6.0
    # view_point[2] += 0

    # process data
    time3 = time.time()
    # print(f'CPU {(time3- time2):.3f} s.')

    try:
        # point_clouds.translate(np.array([0, -shorter_dist, move_z])) # human向Y轴拉近 @shorter_dist, 向上平移 @move_z
        # point_clouds.rotate(rot) #这个仅围绕中心点旋转
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) @ rot.T)
        point_clouds = hidden_point_removal(pcd)
        pcd_new = o3d.geometry.PointCloud.uniform_down_sample(point_clouds, 5)
        # point_clouds = select_points_on_the_scan_line(np.asarray(point_clouds.points))
        # o3d.io.write_point_cloud(save_path, point_clouds)
    except:
        print(f'sample error !!!')

    time4 = time.time()
    final = np.array(pcd_new.points)
    print(final.shape[0])
    return final
    # print(f'{(time4- time3):.3f} s.')
    
def fix_points_num(points, num_points=512):
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

def simulatorLiDAR(file, out_root=None, shorter_dist=0, move_z = 0, rot = np.eye(3), threads = 1):
    
    pose = file['pose']
    trans = file['trans']
    beta = file['shape']
    pc = np.zeros((0,512,3))
    boundary_label = np.zeros((0,512,1))
    for index in range(pose.shape[0]):
        smpl_mesh = smpl(torch.from_numpy(pose[index][np.newaxis,:]).float(), torch.from_numpy(np.zeros((1,10))).float())
        smpl_vertex = smpl_mesh.squeeze().cpu().numpy() + trans[index]
        # save_path = os.path.join(out_root, index+'.pcd')
        sample_pc = sample_data(smpl_vertex, shorter_dist, move_z, rot)
        sample_pc = fix_points_num(sample_pc)
        boundary = outline_onesample(sample_pc)
        pc = np.concatenate((pc, sample_pc[np.newaxis,:]))
        boundary_label = np.concatenate((boundary_label, boundary[np.newaxis,:]))
    return pc, boundary_label
    # if out_root is None:
    #     out_root = root.replace('pose_rot', 'sampled_ouster')
    # os.makedirs(out_root, exist_ok=True)        
    # 
    # filelist = sorted(glob(root+'/*.ply'))
    # 
    # time1 = time.time()
    # 
    # if threads ==  1:
    #     for index in range(0, len(filelist)):
    #         sample_data(filelist[index], out_root, shorter_dist, move_z)
    # 
    # elif threads > 1:
    #     with Pool(threads) as p:
    #         p.map(functools.partial(sample_data, out_root=out_root,
    #               shorter_dist=shorter_dist, move_z=move_z, rot = rot), filelist)
    # else:
    #     print(f'Input threads: {threads} error')
    # 
    # time2 = time.time()
    # 
    # print(f'\n {root} processed. Consumed {(time2- time1):.2f} s.')

import h5py

if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument("--file_path", '-F', type=str,
                        default='')
    parser.add_argument("--out_path", '-O', type=str,
                        default='')
    args = parser.parse_args()
    
    train_set = [
        51801, 51802, 51803, 51808, 51807, 61102, 61103, 61105, 61108, 61109, 61201,
        61202, 61203, 61206, 61207, 61210, 61212, 61213, 61301, 61302, 61304, 61305,
        61306, 61307, 61704, 61705, 61706, 61707, 61708, 61709, 61711, 61714, 61801,
        61802, 61803, 61804, 61805, 61808, 51809, 51804, 51812, 51810
    ]
    test_set = [61107, 61101, 61204, 61209, 61303, 61702, 61713, 61807, 61806, 61106
    ]
    all_set = train_set + test_set
    
    for index in all_set:
        file_path = os.path.join(args.file_path, str(index)+'.hdf5')
        f = h5py.File(file_path,'r')
        out = os.path.join(args.out_path, str(index)+ '.hdf5')
        sample_pc, boundary_label = simulatorLiDAR(f, out_root=out, shorter_dist=0, move_z=0, rot=np.eye(3), threads=1)
        ff = h5py.File(out, 'w')
        ff['boundary_label'] = boundary_label
        ff['sample_pc'] = sample_pc
        ff['full_joints'] = f['full_joints'][:]
        ff['point_clouds'] = f['point_clouds'][:]
        ff['points_num'] = f['points_num'][:]
        ff['pose'] = f['pose'][:]
        ff['rotmats'] = f['rotmats'][:]
        ff['shape'] = f['shape'][:]
        ff['trans'] = f['trans'][:]
        f.close()
        ff.close()
    # for folder in sorted(os.listdir(args.file_path), key=lambda x: int(x)):
    #     if folder not in all_set:
    #         continue
    #         
    #     process_folder = os.path.join(args.file_path, folder)
    #     
    #     # # 模拟生成线扫的激光雷达
    #     if int(folder) in lab_list:
    #         simulatorLiDAR(process_folder, shorter_dist = 5, move_z=4.9-1.85, rot = np.eye(3), threads = 16)
    # 
    #     elif int(folder) in haiyun_list:
    #         simulatorLiDAR(process_folder, shorter_dist = 10, move_z=5.7-1.85, rot = np.eye(3), threads = 16)
    # 
    #     elif int(folder) in haiyun_list_2:
    #         simulatorLiDAR(process_folder, shorter_dist = 5, move_z=6.5-1.85, rot = np.eye(3), threads = 16)
    # 
    #     else:
    #         print(f'No {process_folder}')
