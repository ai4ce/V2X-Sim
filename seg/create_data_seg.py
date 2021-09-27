# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
from nuscenes.nuscenes import NuScenes
import os
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
import time
from data.obj_util import *
from data.config_upperbound import Config, ConfigGlobal
from data.data_util import voxelize_occupy, gen_2d_grid_gt
from PIL import Image

def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    return folder_name

# ---------------------- Extract the scenes, and then pre-process them into BEV maps ----------------------
def create_data(config, nusc, current_agent, config_global, scene_start, scene_end, mode):
    print('prepare dataset', mode)
    channel = 'LIDAR_TOP_id_' + str(current_agent)
    channel_bev = 'BEV_TOP_RAW_id_' + str(current_agent)
    total_sample = 0

    res_scenes = range(100)
    for scene_idx in res_scenes[scene_start:scene_end]:
        curr_scene = nusc.scene[scene_idx]
        first_sample_token = curr_scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)

        # Iterate each sample data
        print("Processing scene {} of agent {} ...".format(scene_idx, current_agent))

        # Added by Yiming, make sure the agent_num equals maximum num (5)
        channel_flag = True
        if channel in curr_sample['data']:
           curr_sample_data = nusc.get('sample_data', curr_sample['data'][channel])
           curr_sample_data_bev = nusc.get('sample_data', curr_sample['data'][channel_bev])
           save_seq_cnt = 0  # only used for save data file name
        else:
           channel_flag = False
           for num_sample in range(100):
               save_directory = check_folder(os.path.join(config.savepath, str(scene_idx+80) + '_' + str(num_sample)))
               save_file_name = os.path.join(save_directory, '0.npy')
               np.save(save_file_name, 0)
               print("  >> Finish sample: {}".format(num_sample))
        # -------------------------------------------------------------------------------------------------
        # ------------------------ We only calculate non-empty channels -----------------------------------
        # -------------------------------------------------------------------------------------------------        
        while channel_flag == True:
            # Get the synchronized point clouds
            if mode == 'upperbound':
                all_pc, all_times, trans_matrices = \
                    LidarPointCloud.from_file_multisweep_upperbound_sample_data(nusc, curr_sample_data, return_trans_matrix=True)
            elif mode == 'lowerbound':
                all_pc, all_times, trans_matrices, target_agent_id, num_sensor = \
                    LidarPointCloud.from_file_multisweep_warp2com_sample_data(current_agent, nusc, curr_sample_data, return_trans_matrix=True)
            else:
                raise NotImplementedError()

            # Store point cloud of each sweep
            pc = all_pc.points

            # Store semantics bev of each agent
            bev_data_path, _, _ = nusc.get_sample_data(curr_sample_data_bev['token'])
            bev_pic = Image.open(bev_data_path)
            bev_pic = bev_pic.resize((256, 256), Image.ANTIALIAS)
            bev_pix = np.array(bev_pic)[:,:,0]
            new_bev_pix = np.zeros(bev_pix.shape)

            new_bev_pix[bev_pix == 10] = 1  ### Vehicles
            new_bev_pix[bev_pix == 6] = 2   ### RoadLine
            new_bev_pix[bev_pix == 14] = 3  ### Ground
            new_bev_pix[bev_pix == 7] = 3   ### Road

            # print(np.unique(bev_pix))

            # Prepare data dictionary for the next step (ie, generating BEV maps)
            save_data_dict = dict()
            save_data_dict['pc_all'] = pc
            save_data_dict['bev_seg'] = new_bev_pix.astype(np.uint8)
            # Now we generate dense and sparse BEV maps
            t = time.time()
            seq_idx = 0
            dense_bev_data = convert_to_dense_bev(save_data_dict, config)
            sparse_bev_data = convert_to_sparse_bev(config, dense_bev_data)

            if mode == 'lowerbound':
                sparse_bev_data['trans_matrices'] = trans_matrices
                sparse_bev_data['target_agent_id'] = target_agent_id
                sparse_bev_data['num_sensor'] = num_sensor

            # save the data
            save_directory = check_folder(os.path.join(config.savepath, str(scene_idx) + '_' + str(save_seq_cnt)))
            save_file_name = os.path.join(save_directory, str(seq_idx) + '.npy')
            np.save(save_file_name, arr=sparse_bev_data)
            total_sample += 1
            print("  >> Finish sample: {}, sequence {} takes {} s".format(save_seq_cnt, seq_idx, time.time()-t))

            save_seq_cnt += 1
            # Skip some keyframes if necessary
            flag = False
            for _ in range(config.num_keyframe_skipped + 1):
                if curr_sample['next'] != '':
                    curr_sample = nusc.get('sample', curr_sample['next'])
                else:
                    flag = True
                    break

            if flag:  # No more keyframes
                break
            else:
                curr_sample_data = nusc.get('sample_data', curr_sample['data'][channel])
                curr_sample_data_bev = nusc.get('sample_data', curr_sample['data'][channel_bev])

# ----------------------------------------------------------------------------------------
# ---------------------- Convert the raw data into (dense) BEV maps ----------------------
# ----------------------------------------------------------------------------------------
def convert_to_dense_bev(seq_data_dict,config):
    data_dict = seq_data_dict
    pc_all = data_dict['pc_all']
    pc_all = pc_all.T

    bev_seg = data_dict['bev_seg']

    # Discretize the input point clouds, and compute the ground-truth displacement vectors
    # The following two variables contain the information for the
    # compact representation of binary voxels, as described in the paper
    voxel_indices_list = list()
    padded_voxel_points_list = list()
    res, voxel_indices = voxelize_occupy(pc_all, voxel_size=config.voxel_size, extents=config.area_extents, return_indices=True)
    voxel_indices_list.append(voxel_indices)
    padded_voxel_points_list.append(res)

    # Compile the batch of voxels, so that they can be fed into the network.
    # Note that, the padded_voxel_points in this script will only be used for sanity check.
    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(np.bool)

    return voxel_indices_list, padded_voxel_points, bev_seg

# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev(config,dense_bev_data):
    save_voxel_indices_list, save_voxel_points, bev_seg = dense_bev_data
    save_voxel_dims = save_voxel_points.shape[1:]
    save_data_dict = dict()

    save_data_dict['bev_seg'] = bev_seg

    for i in range(len(save_voxel_indices_list)):
        save_data_dict['voxel_indices_' + str(i)] = save_voxel_indices_list[i].astype(np.int32)

    # -------------------------------- Sanity Check --------------------------------
    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict['voxel_indices_' + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=np.bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--root', default='/data1/yimingli/carScene-mini', type=str, help='Root path to nuScenes dataset')
    parser.add_argument('-r', '--root', default='/data_1/yml/V2X-Sim', type=str, help='Root path to nuScenes dataset')
    parser.add_argument('-s', '--split', type=str, help='The data split [train/val/test]')
    parser.add_argument('-b', '--scene_start', type=int, help='agent number')
    parser.add_argument('-e', '--scene_end', type=int, help='agent number')
    parser.add_argument('-p', '--savepath', type=str, help='Directory for saving the generated data')
    parser.add_argument('-m', '--mode', type=str, choices=['upperbound', 'lowerbound'])
    args = parser.parse_args()

    nusc = NuScenes(version='v1.0-mini', dataroot=args.root, verbose=True)
    print("Total number of scenes:", len(nusc.scene))
    scene_start = args.scene_start
    scene_end = args.scene_end
    
    root = os.path.join(args.savepath, args.mode, args.split)
    for current_agent in range(5):
        savepath = check_folder(os.path.join(root, 'agent'+str(current_agent)))
        config = Config(args.split, True, savepath=savepath)
        config_global = ConfigGlobal(args.split, True, savepath=savepath)
        create_data(config, nusc, current_agent, config_global, scene_start, scene_end, args.mode)
