from data.obj_util import *
import numpy as np
from data.config import Config, ConfigGlobal
from matplotlib import pyplot as plt
from data.Dataset import V2XSIMDataset
import os
import imageio
import argparse


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


def save_rendered_image(current_scene_idx):
    folder_name = './logs'
    check_folder(folder_name)
    folder_name += '/visualization'
    check_folder(folder_name)
    folder_name += f'/agent{agent_idx}'
    check_folder(folder_name)
    folder_name += f'/{output_dir_name}'
    check_folder(folder_name)
    folder_name += f'/scene_{current_scene_idx}'
    check_folder(folder_name)
    plt.savefig(f'{folder_name}/{idx}.png')


def render_gif():
    print(f'Rendering gif for scene {last_scene_idx}...')

    output_gif_dir = f'./logs/visualization/agent{agent_idx}/{output_dir_name}/scene_{last_scene_idx}'
    output_gif_inner_dir = f'{output_gif_dir}/gif'

    images_path_list = [f.split('.') for f in os.listdir(output_gif_dir) if f.endswith('.png')]
    images_path_list.sort(key=lambda x: int(x[0]))
    images_path_list = [f'{output_gif_dir}/{".".join(file)}' for file in images_path_list]

    ims = [imageio.imread(file) for file in images_path_list]
    check_folder(output_gif_inner_dir)
    output_gif_path = f'{output_gif_inner_dir}/out.gif'
    imageio.mimwrite(output_gif_path, ims, fps=5)

    print(f'Rendered {output_gif_path}')


def visualize():
    gt_max_iou_idx = gt_max_iou[0]['gt_box']

    plt.clf()

    for p in range(data_carscenes.pred_len):

        plt.xlim(0, 256)
        plt.ylim(0, 256)
        for k in range(len(gt_max_iou_idx)):
            anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]
            encode_box = reg_target[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
            decode_box = bev_box_decode_np(encode_box, anchor)
            decode_corner = \
                center_to_corner_box2d(np.asarray([anchor[:2]]), np.asarray([anchor[2:4]]),
                                       np.asarray([anchor[4:]]))[0]

            corners = coor_to_vis(decode_corner, data_carscenes.area_extents, data_carscenes.voxel_size)
            c_x, c_y = np.mean(corners, axis=0)
            corners = np.concatenate([corners, corners[[0]]])

            plt.plot(corners[:, 0], corners[:, 1], c='g', linewidth=2.0, zorder=20)
            plt.scatter(c_x, c_y, s=3, c='g', zorder=20)
            plt.plot([c_x, (corners[1][0] + corners[0][0]) / 2.], [c_y, (corners[1][1] + corners[0][1]) / 2.],
                     linewidth=2.0, c='g', zorder=20)

        occupy = np.max(vis_maps, axis=-1)
        m = np.stack([occupy, occupy, occupy], axis=-1)
        m[m > 0] = 0.99
        occupy = (m * 255).astype(np.uint8)
        # -----------#
        free = np.min(vis_maps, axis=-1)
        m = np.stack([free, free, free], axis=-1)
        m[m < 0] = 0.5
        free = (m * 255).astype(np.uint8)
        # -----------#
        if output_dir_name == 'single_view':
            plt.imshow(np.max(padded_voxel_points.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
        elif output_dir_name == 'multi_view':
            plt.imshow(np.max(padded_voxel_points_teacher.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
        else:
            raise Exception("output_dir_name must be 'single_view' or 'multi_view'")

        if not args.headless:
            plt.pause(0.1)

    current_scene_idx = data_carscenes.seq_scenes[0][idx]
    save_rendered_image(current_scene_idx)

    if current_scene_idx != last_scene_idx or idx == len(data_carscenes) - 1:  # last scene finishes, output gif
        render_gif()
        return True

    return False


if __name__ == "__main__":
    split = 'train'
    config = Config(binary=True, split=split)
    config_global = ConfigGlobal(binary=True, split=split)

    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Whether the images should be rendered in headless mode')
    args = parser.parse_args()

    agent_idx = 4
    data_path = f'/scratch/dm4524/data/V2X-Sim-det/train/agent{agent_idx}'
    # data_path = f'/mnt/NAS/home/yiming/NeurIPS2021-DiscoNet/test/agent{agent_idx}'
    # data_path = '/home/dekunma/CS/ai4ce/multi-agent-perception/det/train_data/agent4'

    for output_dir_name in ('single_view', 'multi_view'):

        data_carscenes = V2XSIMDataset(
            dataset_roots=[data_path], split=split,
            config=config, config_global=config_global, val=True)

        last_scene_idx = data_carscenes.seq_scenes[0][0]

        for idx in range(len(data_carscenes)):
            if idx % 20 == 0:
                print(f'Currently at frame {idx} / {len(data_carscenes)}')

            padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, \
            vis_maps, gt_max_iou, filename, target_agent_id, num_sensor, trans_matrix = data_carscenes[idx][0]

            # anchor_corners_list = get_anchor_corners_list(anchors_map, data_carscenes.box_code_size)
            # anchor_corners_map = anchor_corners_list.reshape(data_carscenes.map_dims[0], data_carscenes.map_dims[1],
            #                                                  len(data_carscenes.anchor_size), 4, 2)

            if visualize():
                # gif rendered. reset scene idx
                last_scene_idx = data_carscenes.seq_scenes[0][idx]

    plt.show()
