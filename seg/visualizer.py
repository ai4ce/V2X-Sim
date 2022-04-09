import argparse
from multiprocessing import Manager

import imageio
from torch.utils.data import Dataset

from data.Dataset_upperbound_seg import V2XSimDataset
from data.config_upperbound import Config
from data.obj_util import *
from utils.obj_class_config import class_to_rgb


class ConfigDet(object):
    def __init__(self, split, binary=True, only_det=True, code_type='faf', loss_type='faf_loss', savepath='', root='',
                 is_cross_road=False):
        self.device = None
        self.split = split
        self.savepath = savepath
        self.binary = binary
        self.only_det = only_det
        self.code_type = code_type
        self.loss_type = loss_type  # corner_loss faf_loss

        # The specifications for BEV maps
        self.voxel_size = (0.25, 0.25, 0.4)
        self.area_extents = np.array([[-32., 32.], [-32., 32.], [-7., -2.]]) if is_cross_road else np.array(
            [[-32., 32.], [-32., 32.], [-3., 2.]])
        self.is_cross_road = is_cross_road
        self.past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
        self.future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
        self.num_past_frames_for_bev_seq = 1  # the number of past frames for BEV map sequence
        self.num_past_pcs = 1  # duplicate self.num_past_frames_for_bev_seq

        self.map_dims = [math.ceil((self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]),
                         math.ceil((self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1]),
                         math.ceil((self.area_extents[2][1] - self.area_extents[2][0]) / self.voxel_size[2])]
        self.only_det = True
        self.root = root
        # debug Data:
        self.code_type = 'faf'
        self.pred_type = 'motion'
        # debug Loss
        self.loss_type = 'corner_loss'
        # debug MGDA
        self.MGDA = False
        # debug when2com
        self.MIMO = True
        # debug Motion Classification
        self.motion_state = False
        self.static_thre = 0.2  # speed lower bound

        # debug use_vis
        self.use_vis = True
        self.use_map = False

        # The specifications for object detection encode
        if self.code_type in ['corner_1', 'corner_2']:
            self.box_code_size = 8  # (\delta{x1},\delta{y1},\delta{x2},\delta{y2},\delta{x3},\delta{y3},\delta{x4},\delta{y4})
        elif self.code_type in ['corner_3']:
            self.box_code_size = 10
        elif self.code_type[0] == 'f':
            self.box_code_size = 6  # (x,y,w,h,sin,cos)
        else:
            print(code_type, ' code type is not implemented yet!')
            exit()

        self.pred_len = 1  # the number of frames for prediction, including the current frame

        # anchor size: (w,h,angle) (according to nuscenes w < h)
        if not self.binary:
            self.anchor_size = np.asarray([[2., 4., 0], [2., 4., math.pi / 2.],
                                           [1., 1., 0], [1., 2., 0.], [1., 2., math.pi / 2.],
                                           [3., 12., 0.], [3., 12., math.pi / 2.]])
        else:
            self.anchor_size = np.asarray([[2., 4., 0], [2., 4., math.pi / 2.],
                                           [2., 4., -math.pi / 4.], [3., 12., 0], [3., 12., math.pi / 2.],
                                           [3., 12., -math.pi / 4.]])

        self.category_threshold = [0.4, 0.4, 0.25, 0.25, 0.4]
        # self.class_map = {'vehicle.audi.a2': 1, 'vehicle.audi.etron': 1, 'vehicle.audi.tt': 1,\
        # 		'vehicle.bmw.grandtourer': 1, 'vehicle.bmw.isetta': 1, 'vehicle.chevrolet.impala': 1,\
        # 		'vehicle.citroen.c3': 1, 'vehicle.dodge_charger.police': 1, 'vehicle.jeep.wrangler_rubicon': 1,\
        # 		'vehicle.lincoln.mkz2017': 1, 'vehicle.mercedes-benz.coupe': 1, 'vehicle.mini.cooperst': 1,\
        # 		'vehicle.mustang.mustang': 1, 'vehicle.nissan.micra': 1, 'vehicle.nissan.patrol': 1,\
        # 		'vehicle.seat.leon': 1, 'vehicle.tesla.cybertruck': 1, 'vehicle.tesla.model3': 1,\
        # 		'vehicle.toyota.prius': 1, 'vehicle.volkswagen.t2': 1, 'vehicle.carlamotors.carlacola': 1,\
        # 		'human.pedestrian': 2, 'vehicle.bh.crossbike': 3, 'vehicle.diamondback.century': 3,\
        # 		'vehicle.gazelle.omafiets': 3, 'vehicle.harley-davidson.low_rider': 3,\
        # 		'vehicle.kawasaki.ninja': 3, 'vehicle.yamaha.yzf': 3}  # background: 0, other: 4
        self.class_map = {'vehicle.car': 1, 'vehicle.emergency.police': 1, 'vehicle.bicycle': 3,
                          'vehicle.motorcycle': 3, 'vehicle.bus.rigid': 2}

        # self.class_map = {'vehicle.car': 1, 'vehicle.truck': 1, 'vehicle.bus': 1, 'human.pedestrian': 2, 'vehicle.bicycle': 3, 'vehicle.motorcycle': 3}  # background: 0, other: 4
        if self.binary:
            self.category_num = 2
        else:
            self.category_num = len(self.category_threshold)
        self.print_feq = 100
        if self.split == 'train':
            self.num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
            self.nsweeps_back = 1  # Number of frames back to the history (including the current timestamp)
            self.nsweeps_forward = 0  # Number of frames into the future (does not include the current timestamp)
            self.skip_frame = 0  # The number of frames skipped for the adjacent sequence
            self.num_adj_seqs = 1  # number of adjacent sequences, among which the time gap is \delta t
        else:
            self.num_keyframe_skipped = 0
            self.nsweeps_back = 1  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
            self.nsweeps_forward = 0
            self.skip_frame = 0
            self.num_adj_seqs = 1

class ConfigGlobalDet(object):
    def __init__(self, split, binary=True, only_det=True, code_type='faf', loss_type='faf_loss', savepath='', root=''):

        self.device = None
        self.split = split
        self.savepath = savepath
        self.binary = binary
        self.only_det = only_det
        self.code_type = code_type
        self.loss_type = loss_type  # corner_loss faf_loss

        # The specifications for BEV maps
        self.voxel_size = (0.25, 0.25, 0.4)
        self.area_extents = np.array([[-96., 96.], [-96., 96.], [-3., 2.]])
        self.past_frame_skip = 0  # when generating the BEV maps, how many history frames need to be skipped
        self.future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
        self.num_past_frames_for_bev_seq = 1  # the number of past frames for BEV map sequence
        self.num_past_pcs = 4  # duplicate self.num_past_frames_for_bev_seq

        self.map_dims = [math.ceil((self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]), \
                         math.ceil((self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1]), \
                         math.ceil((self.area_extents[2][1] - self.area_extents[2][0]) / self.voxel_size[2])]
        self.only_det = True
        self.root = root

        # debug Data:
        self.code_type = 'faf'
        self.pred_type = 'motion'
        # debug Loss
        self.loss_type = 'corner_loss'

        # debug MGDA
        self.MGDA = False
        # debug when2com
        self.MIMO = False
        # debug Motion Classification
        self.motion_state = False
        self.static_thre = 0.2  # speed lower bound

        # debug use_vis
        self.use_vis = True
        self.use_map = False

        # The specifications for object detection encode
        if self.code_type in ['corner_1', 'corner_2']:
            self.box_code_size = 8  # (\delta{x1},\delta{y1},\delta{x2},\delta{y2},\delta{x3},\delta{y3},\delta{x4},\delta{y4})
        elif self.code_type in ['corner_3']:
            self.box_code_size = 10
        elif self.code_type[0] == 'f':
            self.box_code_size = 6  # (x,y,w,h,sin,cos)
        else:
            print(code_type, ' code type is not implemented yet!')
            exit()

        self.pred_len = 1  # the number of frames for prediction, including the current frame

        # anchor size: (w,h,angle) (according to nuscenes w < h)
        if not self.binary:
            self.anchor_size = np.asarray([[2., 4., 0], [2., 4., math.pi / 2.], \
                                           [1., 1., 0], [1., 2., 0.], [1., 2., math.pi / 2.], \
                                           [3., 12., 0.], [3., 12., math.pi / 2.]])
        else:
            self.anchor_size = np.asarray([[2., 4., 0], [2., 4., math.pi / 2.], \
                                           [2., 4., -math.pi / 4.], [3., 12., 0], [3., 12., math.pi / 2.],
                                           [3., 12., -math.pi / 4.]])

        self.category_threshold = [0.4, 0.4, 0.25, 0.25, 0.4]
        self.class_map = {'vehicle.audi.a2': 1, 'vehicle.audi.etron': 1, 'vehicle.audi.tt': 1,
                          'vehicle.bmw.grandtourer': 1, 'vehicle.bmw.isetta': 1, 'vehicle.chevrolet.impala': 1,
                          'vehicle.citroen.c3': 1, 'vehicle.dodge_charger.police': 1,
                          'vehicle.jeep.wrangler_rubicon': 1,
                          'vehicle.lincoln.mkz2017': 1, 'vehicle.mercedes-benz.coupe': 1, 'vehicle.mini.cooperst': 1,
                          'vehicle.mustang.mustang': 1, 'vehicle.nissan.micra': 1, 'vehicle.nissan.patrol': 1,
                          'vehicle.seat.leon': 1, 'vehicle.tesla.cybertruck': 1, 'vehicle.tesla.model3': 1,
                          'vehicle.toyota.prius': 1, 'vehicle.volkswagen.t2': 1, 'vehicle.carlamotors.carlacola': 1,
                          'human.pedestrian': 2, 'vehicle.bh.crossbike': 3, 'vehicle.diamondback.century': 3,
                          'vehicle.gazelle.omafiets': 3, 'vehicle.harley-davidson.low_rider': 3,
                          'vehicle.kawasaki.ninja': 3, 'vehicle.yamaha.yzf': 3}  # background: 0, other: 4
        # self.class_map = {'vehicle.car': 1, 'vehicle.truck': 1, 'vehicle.bus': 1, 'human.pedestrian': 2, 'vehicle.bicycle': 3, 'vehicle.motorcycle': 3}  # background: 0, other: 4
        if self.binary:
            self.category_num = 2
        else:
            self.category_num = len(self.category_threshold)
        self.print_feq = 100
        if self.split == 'train':
            self.num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
            self.nsweeps_back = 1  # Number of frames back to the history (including the current timestamp)
            self.nsweeps_forward = 0  # Number of frames into the future (does not include the current timestamp)
            self.skip_frame = 0  # The number of frames skipped for the adjacent sequence
            self.num_adj_seqs = 1  # number of adjacent sequences, among which the time gap is \delta t
        else:
            self.num_keyframe_skipped = 0
            self.nsweeps_back = 1  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
            self.nsweeps_forward = 0
            self.skip_frame = 0
            self.num_adj_seqs = 1


class NuscenesDataset(Dataset):
    def __init__(self, dataset_root=None, config=None, split=None, cache_size=10000, val=False):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size
        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis
        # dataset_root = dataset_root + '/'+split
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(self.split))
        self.dataset_root = dataset_root
        seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                    if os.path.isdir(os.path.join(self.dataset_root, d))]
        seq_dirs = sorted(seq_dirs)
        self.seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
                          if os.path.isfile(os.path.join(seq_dir, f))]

        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))

        '''
        # For training, the size of dataset should be 17065 * 2; for validation: 1623; for testing: 4309
        if split == 'train' and self.num_sample_seqs != 17065 * 2:
            warnings.warn(">> The size of training dataset is not 17065 * 2.\n")
        elif split == 'val' and self.num_sample_seqs != 1623:
            warnings.warn(">> The size of validation dataset is not 1719.\n")
        elif split == 'test' and self.num_sample_seqs != 4309:
            warnings.warn('>> The size of test dataset is not 4309.\n')
        '''

        # object information
        self.anchors_map = init_anchors_no_check(self.area_extents, self.voxel_size, self.box_code_size,
                                                 self.anchor_size)
        self.map_dims = [int((self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]), \
                         int((self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1])]
        self.reg_target_shape = (
            self.map_dims[0], self.map_dims[1], len(self.anchor_size), self.pred_len, self.box_code_size)
        self.label_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size))
        self.label_one_hot_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size), self.category_num)
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size if split == 'train' else 0
        # self.cache_size = cache_size

    def __len__(self):
        return self.num_sample_seqs

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def __getitem__(self, idx):
        # if idx in self.cache:
        #     gt_dict = self.cache[idx]
        # else:
        seq_file = self.seq_files[idx]
        gt_data_handle = np.load(seq_file, allow_pickle=True)
        gt_dict = gt_data_handle.item()
        if gt_dict == 0:
            return None

        if len(self.cache) < self.cache_size:
            self.cache[idx] = gt_dict

        allocation_mask = gt_dict['allocation_mask'].astype(np.bool)
        reg_loss_mask = gt_dict['reg_loss_mask'].astype(np.bool)
        gt_max_iou = gt_dict['gt_max_iou']
        motion_one_hot = np.zeros(5)
        motion_mask = np.zeros(5)

        # load regression target
        reg_target_sparse = gt_dict['reg_target_sparse']
        # need to be modified Yiqi , only use reg_target and allocation_map
        reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

        reg_target[allocation_mask] = reg_target_sparse
        reg_target[np.bitwise_not(reg_loss_mask)] = 0
        label_sparse = gt_dict['label_sparse']

        one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
        label_one_hot = np.zeros(self.label_one_hot_shape)
        label_one_hot[:, :, :, 0] = 1
        label_one_hot[allocation_mask] = one_hot_label_sparse

        if self.config.motion_state:
            motion_sparse = gt_dict['motion_state']
            motion_one_hot_label_sparse = self.get_one_hot(motion_sparse, 3)
            motion_one_hot = np.zeros(self.label_one_hot_shape[:-1] + (3,))
            motion_one_hot[:, :, :, 0] = 1
            motion_one_hot[allocation_mask] = motion_one_hot_label_sparse
            motion_mask = (motion_one_hot[:, :, :, 2] == 1)

        if self.only_det:
            reg_target = reg_target[:, :, :, :1]
            reg_loss_mask = reg_loss_mask[:, :, :, :1]

        # only center for pred

        elif self.config.pred_type in ['motion', 'center']:
            reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
            reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
            reg_loss_mask[:, :, :, 1:, 2:] = False

        if self.config.use_map:
            if ('map_allocation_0' in gt_dict.keys()) or ('map_allocation' in gt_dict.keys()):
                semantic_maps = []
                for m_id in range(self.config.map_channel):
                    map_alloc = gt_dict['map_allocation_' + str(m_id)]
                    map_sparse = gt_dict['map_sparse_' + str(m_id)]
                    recover = np.zeros(tuple(self.config.map_dims[:2]))
                    recover[map_alloc] = map_sparse
                    recover = np.rot90(recover, 3)
                    # recover_map = cv2.resize(recover,(self.config.map_dims[0],self.config.map_dims[1]))
                    semantic_maps.append(recover)
                semantic_maps = np.asarray(semantic_maps)
        else:
            semantic_maps = np.zeros(0)
        '''
        if self.binary:
            reg_target = np.concatenate([reg_target[:,:,:2],reg_target[:,:,5:]],axis=2)
            reg_loss_mask = np.concatenate([reg_loss_mask[:,:,:2],reg_loss_mask[:,:,5:]],axis=2)
            label_one_hot = np.concatenate([label_one_hot[:,:,:2],label_one_hot[:,:,5:]],axis=2)

        '''
        padded_voxel_points = list()

        for i in range(self.num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            curr_voxels = np.zeros(self.dims, dtype=bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            curr_voxels = np.rot90(curr_voxels, 3)
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)
        anchors_map = self.anchors_map
        '''
        if self.binary:
            anchors_map = np.concatenate([anchors_map[:,:,:2],anchors_map[:,:,5:]],axis=2)
        '''
        if self.config.use_vis:
            vis_maps = np.zeros(
                (self.num_past_pcs, self.config.map_dims[-1], self.config.map_dims[0], self.config.map_dims[1]))
            vis_free_indices = gt_dict['vis_free_indices']
            vis_occupy_indices = gt_dict['vis_occupy_indices']
            vis_maps[vis_occupy_indices[0, :], vis_occupy_indices[1, :], vis_occupy_indices[2, :], vis_occupy_indices[3,
                                                                                                   :]] = math.log(
                0.7 / (1 - 0.7))
            vis_maps[vis_free_indices[0, :], vis_free_indices[1, :], vis_free_indices[2, :], vis_free_indices[3,
                                                                                             :]] = math.log(
                0.4 / (1 - 0.4))
            vis_maps = np.swapaxes(vis_maps, 2, 3)
            vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
            for v_id in range(vis_maps.shape[0]):
                vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
            vis_maps = vis_maps[-1]

        else:
            vis_maps = np.zeros(0)

        padded_voxel_points = padded_voxel_points.astype(np.float32)
        label_one_hot = label_one_hot.astype(np.float32)
        reg_target = reg_target.astype(np.float32)
        anchors_map = anchors_map.astype(np.float32)
        motion_one_hot = motion_one_hot.astype(np.float32)
        semantic_maps = semantic_maps.astype(np.float32)
        vis_maps = vis_maps.astype(np.float32)

        if self.val:
            return padded_voxel_points, label_one_hot, \
                   reg_target, reg_loss_mask, anchors_map, motion_one_hot, motion_mask, vis_maps, [
                       {"gt_box": gt_max_iou}], [seq_file]
        else:
            return padded_voxel_points, label_one_hot, \
                   reg_target, reg_loss_mask, anchors_map, motion_one_hot, motion_mask, vis_maps



class V2XSIMDatasetDet(Dataset):
    def __init__(self, dataset_roots=None, config=None, config_global=None, split=None, cache_size=10000, val=False, no_cross_road=False):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        category_num: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if split is None:
            self.split = config.split
        else:
            self.split = split
        self.voxel_size = config.voxel_size
        self.area_extents = config.area_extents
        self.category_num = config.category_num
        self.future_frame_skip = config.future_frame_skip
        self.pred_len = config.pred_len
        self.box_code_size = config.box_code_size
        self.anchor_size = config.anchor_size

        self.val = val
        self.only_det = config.only_det
        self.binary = config.binary
        self.config = config
        self.use_vis = config.use_vis

        self.no_cross_road = no_cross_road
        self.padded_voxel_points_meta = None
        self.label_one_hot_meta = None
        self.reg_target_meta = None
        self.reg_loss_mask_meta = None
        self.anchors_map_meta = None
        self.vis_maps_meta = None

        # dataset_root = dataset_root + '/'+split
        if dataset_roots is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(self.split))
        self.dataset_roots = dataset_roots
        self.num_agent = len(dataset_roots)
        self.seq_files = []
        self.seq_scenes = []
        for dataset_root in self.dataset_roots:
            # sort directories
            dir_list = [d.split('_') for d in os.listdir(dataset_root)]
            dir_list.sort(key=lambda x: (int(x[0]), int(x[1])))
            self.seq_scenes.append([int(s[0]) for s in dir_list])  # which scene this frame belongs to (required for visualization)
            dir_list = ['_'.join(x) for x in dir_list]

            seq_dirs = [os.path.join(dataset_root, d) for d in dir_list
                        if os.path.isdir(os.path.join(dataset_root, d))]

            self.seq_files.append([os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
                                   if os.path.isfile(os.path.join(seq_dir, f))])

        self.num_sample_seqs = len(self.seq_files[0])
        print("The number of {} sequences: {}".format(self.split, self.num_sample_seqs))
        # object information
        self.anchors_map = init_anchors_no_check(self.area_extents, self.voxel_size, self.box_code_size,
                                                 self.anchor_size)
        self.map_dims = [int((self.area_extents[0][1] - self.area_extents[0][0]) / self.voxel_size[0]),
                         int((self.area_extents[1][1] - self.area_extents[1][0]) / self.voxel_size[1])]
        self.reg_target_shape = (
            self.map_dims[0], self.map_dims[1], len(self.anchor_size), self.pred_len, self.box_code_size)
        self.label_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size))
        self.label_one_hot_shape = (self.map_dims[0], self.map_dims[1], len(self.anchor_size), self.category_num)
        self.dims = config.map_dims
        self.num_past_pcs = config.num_past_pcs
        manager = Manager()
        self.cache = [manager.dict() for _ in range(self.num_agent)]
        self.cache_size = cache_size if split == 'train' else 0

        if self.val:
            self.voxel_size_global = config_global.voxel_size
            self.area_extents_global = config_global.area_extents
            self.pred_len_global = config_global.pred_len
            self.box_code_size_global = config_global.box_code_size
            self.anchor_size_global = config_global.anchor_size
            # object information
            self.anchors_map_global = init_anchors_no_check(self.area_extents_global, self.voxel_size_global,
                                                            self.box_code_size_global, self.anchor_size_global)
            self.map_dims_global = [
                int((self.area_extents_global[0][1] - self.area_extents_global[0][0]) / self.voxel_size_global[0]), \
                int((self.area_extents_global[1][1] - self.area_extents_global[1][0]) / self.voxel_size_global[1])]
            self.reg_target_shape_global = (
                self.map_dims_global[0], self.map_dims_global[1], len(self.anchor_size_global), self.pred_len_global,
                self.box_code_size_global)
            self.dims_global = config_global.map_dims
        self.get_meta()

    def get_meta(self):
        meta = NuscenesDataset(dataset_root=self.dataset_roots[0], split=self.split, config=self.config, val=self.val)
        if not self.val:
            self.padded_voxel_points_meta, self.label_one_hot_meta, self.reg_target_meta, self.reg_loss_mask_meta, \
            self.anchors_map_meta, _, _, self.vis_maps_meta = meta[0]
        else:
            meta_0 = meta[0]
            # empty
            if meta_0 == None:
                return
            
            self.padded_voxel_points_meta, self.label_one_hot_meta, self.reg_target_meta, self.reg_loss_mask_meta, \
            self.anchors_map_meta, _, _, self.vis_maps_meta, _, _ = meta[0]
        del meta

    def __len__(self):
        return self.num_sample_seqs

    def get_one_hot(self, label, category_num):
        one_hot_label = np.zeros((label.shape[0], category_num))
        for i in range(label.shape[0]):
            one_hot_label[i][label[i]] = 1

        return one_hot_label

    def pick_single_agent(self, agent_id, idx):
        empty_flag = False
        if idx in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][idx]
        else:
            seq_file = self.seq_files[agent_id][idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                padded_voxel_points = np.zeros_like(self.padded_voxel_points_meta)
                label_one_hot = np.zeros_like(self.label_one_hot_meta)
                reg_target = np.zeros_like(self.reg_target_meta)
                anchors_map = np.zeros_like(self.anchors_map_meta)
                vis_maps = np.zeros_like(self.vis_maps_meta)
                reg_loss_mask = np.zeros_like(self.reg_loss_mask_meta)
                if self.val:
                    return padded_voxel_points, padded_voxel_points, label_one_hot, \
                           reg_target, reg_loss_mask, anchors_map, vis_maps, [
                               {"gt_box": [[0, 0, 0, 0], [0, 0, 0, 0]]}], [seq_file], \
                           0, 0, np.zeros((5, 4, 4))
                else:
                    return padded_voxel_points, padded_voxel_points, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, 0, 0, np.zeros(
                        (5, 4, 4))
            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][idx] = gt_dict

        if empty_flag == False:
            allocation_mask = gt_dict['allocation_mask'].astype(bool)
            reg_loss_mask = gt_dict['reg_loss_mask'].astype(bool)
            gt_max_iou = gt_dict['gt_max_iou']

            # load regression target
            reg_target_sparse = gt_dict['reg_target_sparse']
            # need to be modified Yiqi , only use reg_target and allocation_map
            reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

            reg_target[allocation_mask] = reg_target_sparse
            reg_target[np.bitwise_not(reg_loss_mask)] = 0
            label_sparse = gt_dict['label_sparse']

            one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
            label_one_hot = np.zeros(self.label_one_hot_shape)
            label_one_hot[:, :, :, 0] = 1
            label_one_hot[allocation_mask] = one_hot_label_sparse

            if self.only_det:
                reg_target = reg_target[:, :, :, :1]
                reg_loss_mask = reg_loss_mask[:, :, :, :1]

            # only center for pred
            elif self.config.pred_type in ['motion', 'center']:
                reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
                reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
                reg_loss_mask[:, :, :, 1:, 2:] = False

            padded_voxel_points = list()

            for i in range(self.num_past_pcs):
                indices = gt_dict['voxel_indices_' + str(i)]
                curr_voxels = np.zeros(self.dims, dtype=bool)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                curr_voxels = np.rot90(curr_voxels, 3)
                padded_voxel_points.append(curr_voxels)
            padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)
            anchors_map = self.anchors_map

            if self.config.use_vis:
                vis_maps = np.zeros(
                    (self.num_past_pcs, self.config.map_dims[-1], self.config.map_dims[0], self.config.map_dims[1]))
                vis_free_indices = gt_dict['vis_free_indices']
                vis_occupy_indices = gt_dict['vis_occupy_indices']
                vis_maps[
                    vis_occupy_indices[0, :], vis_occupy_indices[1, :], vis_occupy_indices[2, :], vis_occupy_indices[3,
                                                                                                  :]] = math.log(
                    0.7 / (1 - 0.7))
                vis_maps[vis_free_indices[0, :], vis_free_indices[1, :], vis_free_indices[2, :], vis_free_indices[3,
                                                                                                 :]] = math.log(
                    0.4 / (1 - 0.4))
                vis_maps = np.swapaxes(vis_maps, 2, 3)
                vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
                for v_id in range(vis_maps.shape[0]):
                    vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
                vis_maps = vis_maps[-1]
            else:
                vis_maps = np.zeros(0)

            trans_matrices = gt_dict['trans_matrices']

            padded_voxel_points = padded_voxel_points.astype(np.float32)
            label_one_hot = label_one_hot.astype(np.float32)
            reg_target = reg_target.astype(np.float32)
            anchors_map = anchors_map.astype(np.float32)
            vis_maps = vis_maps.astype(np.float32)

            target_agent_id = gt_dict['target_agent_id']
            num_sensor = gt_dict['num_sensor']

            if 'voxel_indices_teacher' in gt_dict:

                padded_voxel_points_teacher = list()
                if self.no_cross_road:
                    indices_teacher = gt_dict['voxel_indices_teacher_no_cross_road']
                else:
                    indices_teacher = gt_dict['voxel_indices_teacher']

                curr_voxels_teacher = np.zeros(self.dims, dtype=bool)
                curr_voxels_teacher[indices_teacher[:, 0], indices_teacher[:, 1], indices_teacher[:, 2]] = 1
                curr_voxels_teacher = np.rot90(curr_voxels_teacher, 3)
                padded_voxel_points_teacher.append(curr_voxels_teacher)
                padded_voxel_points_teacher = np.stack(padded_voxel_points_teacher, 0).astype(np.float32)
                padded_voxel_points_teacher = padded_voxel_points_teacher.astype(np.float32)
            else:  # TODO upperbound eval in old
                padded_voxel_points_teacher = padded_voxel_points

            if self.val:
                return padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, [
                    {"gt_box": gt_max_iou}], [seq_file], \
                       target_agent_id, num_sensor, trans_matrices


            else:
                return padded_voxel_points, padded_voxel_points_teacher, label_one_hot, reg_target, reg_loss_mask, anchors_map, vis_maps, \
                       target_agent_id, num_sensor, trans_matrices

    def __getitem__(self, idx):
        res = []
        for i in range(self.num_agent):
            res.append(self.pick_single_agent(i, idx))
        return res


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


def save_rendered_image():
    folder_name = './logs'
    check_folder(folder_name)
    folder_name += '/visualization'
    check_folder(folder_name)
    folder_name += f'/agent{agent_idx}'
    check_folder(folder_name)
    folder_name += f'/scene_{current_scene_idx}'
    check_folder(folder_name)
    plt.savefig(f'{folder_name}/{idx}.png', bbox_inches='tight')

def render_gif():

    output_gif_dir = f'./logs/visualization/agent{agent_idx}/scene_{last_scene_idx}'
    # if no image output for the last scene
    if not os.path.exists(output_gif_dir):
        return

    print(f'Rendering gif for scene {last_scene_idx}...')
    output_gif_inner_dir = f'{output_gif_dir}/gif'

    images_path_list = [f.split('.') for f in os.listdir(output_gif_dir) if f.endswith('.png')]
    images_path_list.sort(key=lambda x: int(x[0]))
    images_path_list = [f'{output_gif_dir}/{".".join(file)}' for file in images_path_list]

    ims = [imageio.imread(file) for file in images_path_list]
    check_folder(output_gif_inner_dir)
    output_gif_path = f'{output_gif_inner_dir}/out.gif'
    imageio.mimwrite(output_gif_path, ims, fps=5)

    print(f'Rendered {output_gif_path}')

def set_title_for_axes(ax, title):
    ax.title.set_text(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_idx', default=0, type=int, help='which agent')
    parser.add_argument('--split', default='val', type=str)
    args = parser.parse_args()

    split = 'train'
    config = Config(binary=True, split=split)

    agent_idx = args.agent_idx
    data_path = f'/scratch/dm4524/data/V2X-Sim-seg/{args.split}/agent{agent_idx}'
    data_path_det = f'/scratch/dm4524/data/V2X-Sim-det/{args.split}/agent{agent_idx}'

    data_carscenes = V2XSimDataset(dataset_roots=[data_path],
                                   split=split, config=config, val=True, kd_flag=True, bound='lowerbound', no_cross_road=False)
    
    data_carscenes_no_cross_road = V2XSimDataset(dataset_roots=[data_path],
                                   split=split, config=config, val=True, kd_flag=True, bound='lowerbound', no_cross_road=True)

    config_global = ConfigGlobalDet(binary=True, split=split)
    configDet = ConfigDet(binary=True, split=split)
    data_carscenes_det = V2XSIMDatasetDet(
            dataset_roots=[data_path_det], split=split, config_global=config_global,
            config=configDet, val=True)
    data_carscenes_det_no_cross_road = V2XSIMDatasetDet(
            dataset_roots=[data_path_det], split=split, config_global=config_global,
            config=configDet, val=True, no_cross_road=True)

    last_scene_idx = data_carscenes.seq_scenes[0][0]
    for idx in range(7700, 7801):
        if idx % 20 == 0:
            print(f'Currently at frame {idx} / {len(data_carscenes)}')

        padded_voxel_points, padded_voxel_points_teacher, seg_bev = data_carscenes[idx][0]
        padded_voxel_points_ncr, padded_voxel_points_teacher_ncr, seg_bev = data_carscenes_no_cross_road[idx][0]


        padded_voxel_points_det, padded_voxel_points_teacher_det, label_one_hot, reg_target, reg_loss_mask, anchors_map, \
            vis_maps, gt_max_iou, filename, target_agent_id, num_sensor, trans_matrix = data_carscenes_det[idx][0]
        
        # empty
        if num_sensor == 0:
            continue
        
        padded_voxel_points_det_ncr, padded_voxel_points_teacher_det_ncr, label_one_hot, reg_target, reg_loss_mask, _, \
            vis_maps, gt_max_iou, filename, target_agent_id, num_sensor, trans_matrix = data_carscenes_det_no_cross_road[idx][0]

        plt.clf()
        for p in range(data_carscenes.pred_len):
            plt.xlim(0, 256)
            plt.ylim(0, 256)

            # just put it here so that we know the ratio is adjustable
            gs_kw = dict(width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])
            fig, axes = plt.subplot_mosaic([
                ['a', 'b', 'c', 'seg1'],
                ['d', 'e', 'f', 'seg2'] # seg 1 and seg 2 are the same
            ],
                gridspec_kw=gs_kw, figsize=(8, 5),
                constrained_layout=True)
            
            plt.axis('off')

            # convert tensors to run on CPU
            padded_voxel_points = padded_voxel_points.cpu().detach().numpy()
            padded_voxel_points_teacher = padded_voxel_points_teacher.cpu().detach().numpy()
            padded_voxel_points_teacher_ncr = padded_voxel_points_teacher_ncr.cpu().detach().numpy()

            axes['a'].imshow(np.max(padded_voxel_points.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            set_title_for_axes(axes['a'], 'seg single-view')

            axes['b'].imshow(np.max(padded_voxel_points_teacher.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            set_title_for_axes(axes['b'], 'seg multi-view\n w/ cross')

            axes['c'].imshow(np.max(padded_voxel_points_teacher_ncr.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            set_title_for_axes(axes['c'], 'seg multi-view\n w/o cross')

            # ===== det
            def draw_gt_box(ax):
                gt_max_iou_idx = gt_max_iou[0]['gt_box']
                for p in range(data_carscenes_det.pred_len):

                    for k in range(len(gt_max_iou_idx)):
                        anchor = anchors_map[tuple(gt_max_iou_idx[k][:-1])]
                        encode_box = reg_target[tuple(gt_max_iou_idx[k][:-1]) + (p,)]
                        decode_box = bev_box_decode_np(encode_box, anchor)
                        decode_corner = \
                            center_to_corner_box2d(np.asarray([anchor[:2]]), np.asarray([anchor[2:4]]),
                                                   np.asarray([anchor[4:]]))[0]

                        corners = coor_to_vis(decode_corner, data_carscenes_det.area_extents, data_carscenes_det.voxel_size)

                        c_x, c_y = np.mean(corners, axis=0)
                        corners = np.concatenate([corners, corners[[0]]])

                        ax.plot(corners[:, 0], corners[:, 1], c='g', linewidth=2.0, zorder=20)
                        ax.scatter(c_x, c_y, s=3, c='g', zorder=20)
                        ax.plot([c_x, (corners[1][0] + corners[0][0]) / 2.], [c_y, (corners[1][1] + corners[0][1]) / 2.],
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
                    #---------------

            draw_gt_box(axes['d'])
            axes['d'].imshow(np.max(padded_voxel_points_det.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            set_title_for_axes(axes['d'], 'det single-view')

            draw_gt_box(axes['e'])
            axes['e'].imshow(np.max(padded_voxel_points_teacher_det.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            set_title_for_axes(axes['e'], 'det multi-view\n w/ cross')
            #========

            draw_gt_box(axes['f'])
            axes['f'].imshow(np.max(padded_voxel_points_teacher_det_ncr.reshape(256, 256, 13), axis=2), alpha=1.0, zorder=12)
            set_title_for_axes(axes['f'], 'det muti-view\n w/o cross')

            seg_image = np.zeros(shape=(256, 256, 3), dtype=np.dtype("uint8"))

            for k in ('seg1', 'seg2'):
                for key, value in class_to_rgb.items():
                    seg_image[np.where(seg_bev == key)] = value
                axes[k].imshow(seg_image)
                set_title_for_axes(axes[k], 'seg')

            current_scene_idx = data_carscenes.seq_scenes[0][idx]

            save_rendered_image()
            if current_scene_idx != last_scene_idx or idx == len(data_carscenes) - 1:  # last scene finishes, output gif
                render_gif()
                last_scene_idx = current_scene_idx

            plt.close('all')

    if data_carscenes.seq_scenes[0][idx] != last_scene_idx or idx == len(data_carscenes) - 1:  # last scene finishes, output gif
        render_gif()
        last_scene_idx = data_carscenes.seq_scenes[0][idx]
