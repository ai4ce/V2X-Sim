# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.
import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

from nuscenes.utils.geometry_utils import view_points, transform_matrix

class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: np.ndarray):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(), 'Error: Pointcloud points must have format: %d x n' % self.nbr_dims()
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'PointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times

    @classmethod
    def from_file_multisweep_bf_sample_data(cls,
                                            nusc: 'NuScenes',
                                            ref_sd_rec: Dict,
                                            nsweeps_back: int = 5,
                                            nsweeps_forward: int = 5,
                                            return_trans_matrix: bool = False,
                                            min_distance: float = 1.0):
        """
        Return a point cloud that aggregates multiple sweeps. The sweeps trace back as well as into the future.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        For this function, the reference sweep is supposed to be from sample data record (not sample. ie, keyframe).
        :param nusc: A NuScenes instance.
        :param ref_sd_rec: The current sample data record.
        :param nsweeps_back: Number of sweeps to aggregate. The sweeps trace back.
        :param nsweeps_forward: Number of sweeps to aggregate. The sweeps are obtained from the future.
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        skip_frame = 0

        for k in range(nsweeps_back):
            # Load up the pointcloud.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))

#            current_pc.points[[0,1], :] = current_pc.points[[1,0], :]

#            current_pc.points[0, :] = - current_pc.points[0, :]

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            current_pc.transform(trans_matrix)
            
            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference

            if k % (skip_frame + 1) == 0:
                times = time_lag * np.ones((1, current_pc.nbr_points()))
            else:
                times = time_lag * np.ones((1, 1))  # dummy value
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            if k % (skip_frame + 1) == 0:
                all_pc.points = np.hstack((all_pc.points, current_pc.points))
            else:
                tmp_points = np.zeros((4, 1), dtype=np.float32)
                all_pc.points = np.hstack((all_pc.points, tmp_points))  # dummy value

            if current_sd_rec['prev'] == '':  # Abort if there are no previous sweeps.
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        # Aggregate the future sweeps
        current_sd_rec = ref_sd_rec

#        # Abort if there are no future sweeps. Return.
#        if current_sd_rec['next'] == '':
#            return all_pc, np.squeeze(all_times, axis=0)
#        else:
#            current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        for k in range(1, nsweeps_forward + 1):
            # Load up the pointcloud.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))

            # Get the pose for this future sweep.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # negative difference

            if k % (skip_frame + 1) == 0:
                times = time_lag * np.ones((1, current_pc.nbr_points()))
            else:
                times = time_lag * np.ones((1, 1))  # dummy value
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            if k % (skip_frame + 1) == 0:
                all_pc.points = np.hstack((all_pc.points, current_pc.points))
            else:
                tmp_points = np.zeros((4, 1), dtype=np.float32)
                all_pc.points = np.hstack((all_pc.points, tmp_points))  # dummy value

            if current_sd_rec['next'] == '':  # Abort if there are no future sweeps.
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        if return_trans_matrix:
            return all_pc, np.squeeze(all_times, 0), trans_matrix_list
        else:
            return all_pc, np.squeeze(all_times, 0)

    @classmethod
    def from_file_multisweep_upperbound_sample_data(cls,
                                                    nusc: 'NuScenes',
                                                    ref_sd_rec: Dict,
                                                    return_trans_matrix: bool = False,
                                                    min_distance: float = 1.0):
        """
        Added by Yiming. 2021.4.14 teacher's input
        Upperbound dataloader: transform the sweeps into the local coordinate of agent 0,
        :param ref_sd_rec: The current sample data record (lidar_top_id_0)
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        skip_frame = 0
        sample_record = nusc.get('sample', ref_sd_rec['sample_token'])
        num_sensor = int(len(sample_record['data']) / 2)

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            current_pc.transform(trans_matrix)

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * (current_sd_rec['timestamp'] + k)  # positive difference

            if k % (skip_frame + 1) == 0:
                times = time_lag * np.ones((1, current_pc.nbr_points()))
            else:
                times = time_lag * np.ones((1, 1))  # dummy value

            all_times = np.hstack((all_times, times))
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        if return_trans_matrix:
            return all_pc, np.squeeze(all_times, 0), trans_matrix_list
        else:
            return all_pc, np.squeeze(all_times, 0)

    @classmethod
    def from_file_multisweep_warp2com_sample_data(cls,
                                         num_agent,
                                         nusc: 'NuScenes',
                                         ref_sd_rec: Dict,
                                         return_trans_matrix: bool = False,
                                         min_distance: float = 1.0):
        """
        Added by Yiming. 2021/3/27
        V2V dataloader: calculate relative pose and overlap mask between different agents
        :param ref_sd_rec: The current sample data record
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        sample_record = nusc.get('sample', ref_sd_rec['sample_token'])
        num_sensor = int(len(sample_record['data'])/2)
        x_global = [[] for i in range(num_sensor)]
        y_global = [[] for i in range(num_sensor)]
        x_local = [[] for i in range(num_sensor)]
        y_local = [[] for i in range(num_sensor)]

        # the following two lists will store the data for each agent
        pc_for_each_agent = list()
        times_for_each_agent = list()
        target_agent_id = None  # which agent is the center agent

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])

            y_global[k] = current_pose_rec['translation'][1]
            x_global[k] = current_pose_rec['translation'][0]

        for k in range(num_sensor):
            x_local[k] = x_global[k] - x_global[num_agent]
            y_local[k] = y_global[k] - y_global[num_agent]

        for k in range(num_sensor):

            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)

            if x_local[k] == 0.0 and y_local[k] == 0.0:
               target_agent_id = k
               current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
               # Remove close points and add timevector.
               current_pc.remove_close(min_distance)
               time_lag = ref_time - 1e-6 * (current_sd_rec['timestamp'])  # positive difference
               times = time_lag * np.ones((1, current_pc.nbr_points()))
               all_times = np.hstack((all_times, times))
            
            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(np.sum([current_pose_rec['translation']],axis=0),
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

        for k in range(5-num_sensor):
            trans_matrix_list.append(np.zeros((4, 4)))

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        if return_trans_matrix:
            return current_pc, np.squeeze(all_times, 0), trans_matrix_list, target_agent_id, num_sensor
        else:
            return current_pc, np.squeeze(all_times, 0), target_agent_id, num_sensor


    @classmethod
    def from_file_multisweep_v2v_sample_data(cls,
                                         num_agent,
                                         nusc: 'NuScenes',
                                         ref_sd_rec: Dict,
                                         return_trans_matrix: bool = False,
                                         min_distance: float = 1.0):
        """
        Added by Yiming. 2020/9/25
        V2V dataloader: calculate relative pose and overlap mask between different agents
        :param ref_sd_rec: The current sample data record
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        sample_record = nusc.get('sample', ref_sd_rec['sample_token'])
        num_sensor = int(len(sample_record['data'])/2)
        x_global = [[] for i in range(num_sensor)]
        y_global = [[] for i in range(num_sensor)]
        x_local = [[] for i in range(num_sensor)]
        y_local = [[] for i in range(num_sensor)]

        # the following two lists will store the data for each agent
        pc_for_each_agent = list()
        times_for_each_agent = list()
        target_agent_id = None  # which agent is the center agent

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])

            y_global[k] = current_pose_rec['translation'][1]
            x_global[k] = current_pose_rec['translation'][0]

        for k in range(num_sensor):
            x_local[k] = x_global[k] - x_global[num_agent]
            y_local[k] = y_global[k] - y_global[num_agent]

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            
            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(np.sum([current_pose_rec['translation']],axis=0),
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            current_pc.transform(trans_matrix)

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * (current_sd_rec['timestamp'])  # positive difference

            pc_for_each_agent.append(current_pc)

            times = time_lag * np.ones((1, current_pc.nbr_points()))
            times_for_each_agent.append(np.squeeze(times, 0))

            if x_local[k] == 0.0 and y_local[k] == 0.0:
                target_agent_id = k

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)
        assert target_agent_id is not None, "The target_agent_id is None."

        if return_trans_matrix:
            return pc_for_each_agent, times_for_each_agent, trans_matrix_list, num_sensor, x_local, y_local, target_agent_id
        else:
            return pc_for_each_agent, times_for_each_agent, num_sensor, x_local, y_local, target_agent_id

    @classmethod
    def from_file_multisweep_consensus_sample_data(cls,
                                             num_agent,
                                             nusc: 'NuScenes',
                                             ref_sd_rec: Dict,
                                             return_trans_matrix: bool = False,
                                             min_distance: float = 1.0):
        """
        Added by Yiming. 2020/9/25
        V2V dataloader: calculate relative pose and overlap mask between different agents
        :param ref_sd_rec: The current sample data record
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        sample_record = nusc.get('sample', ref_sd_rec['sample_token'])
        num_sensor = int(len(sample_record['data']) / 2)
        x_global = [[] for i in range(num_sensor)]
        y_global = [[] for i in range(num_sensor)]
        x_local = [[] for i in range(num_sensor)]
        y_local = [[] for i in range(num_sensor)]

        # the following two lists will store the data for each agent
        pc_for_each_agent = list()
        times_for_each_agent = list()
        target_agent_id = None  # which agent is the center agent

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])

            y_global[k] = current_pose_rec['translation'][1]
            x_global[k] = current_pose_rec['translation'][0]

        for k in range(num_sensor):
            x_local[k] = x_global[k] - x_global[num_agent]
            y_local[k] = y_global[k] - y_global[num_agent]

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(np.sum([current_pose_rec['translation']], axis=0),
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            current_pc.transform(trans_matrix)

            trans_matrix[:3, 3] = 4 * trans_matrix[:3, 3]

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * (current_sd_rec['timestamp'])  # positive difference

            pc_for_each_agent.append(current_pc)

            times = time_lag * np.ones((1, current_pc.nbr_points()))
            times_for_each_agent.append(np.squeeze(times, 0))

            if x_local[k] == 0.0 and y_local[k] == 0.0:
                target_agent_id = k

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)
        assert target_agent_id is not None, "The target_agent_id is None."

        if return_trans_matrix:
            return pc_for_each_agent, times_for_each_agent, trans_matrix_list, num_sensor, x_local, y_local, target_agent_id
        else:
            return pc_for_each_agent, times_for_each_agent, num_sensor, x_local, y_local, target_agent_id

    @classmethod
    def from_file_multisweep_globalmap_sample_data(cls,
                                         x_center,
                                         y_center,
                                         nusc: 'NuScenes',
                                         ref_sd_rec: Dict,
                                         return_trans_matrix: bool = False,
                                         flag_init: bool = False,
                                         min_distance: float = 1.0):
        """
        Added by Yiming. 2020/10/7
        when2com/v2v dataloader for global mAP evaluation: transform the sweeps into the local coordinate of agent 0, 
        then translating to a fixed point (x_center, y_center)
        :param x_center, y_center: centre location of multiple lidar points
        :param ref_sd_rec: The current sample data record (lidar_top_id_0)
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        sample_record = nusc.get('sample', ref_sd_rec['sample_token'])
        lidar_top_id_0_token = sample_record['data']['LIDAR_TOP_id_0']
        ref_sd_rec = nusc.get('sample_data', lidar_top_id_0_token)

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate multiple agents' sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        skip_frame = 0
        num_sensor = int(len(sample_record['data'])/2)
        x_ego = [[] for i in range(num_sensor)]
        y_ego = [[] for i in range(num_sensor)]

        x_filter = [[] for i in range(num_sensor)]
        y_filter = [[] for i in range(num_sensor)]

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])

            y_ego[k] = current_pose_rec['translation'][1]
            x_ego[k] = current_pose_rec['translation'][0]

            x_filter[k] = x_ego[k] - x_ego[0]
            y_filter[k] = y_ego[k] - y_ego[0]

        if flag_init == False:
           x_center = sum(x_filter)/num_sensor
           y_center = sum(y_filter)/num_sensor

        translation_vector = [-x_center, -y_center, 0]

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            
            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(np.sum([current_pose_rec['translation'],translation_vector],axis=0),
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)

            all_pc.points = np.hstack((all_pc.points, current_pc.points))

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        if return_trans_matrix:
            return all_pc, trans_matrix_list, x_center, y_center, x_ego, y_ego, num_sensor

        else:
            return all_pc, x_center, y_center, x_ego, y_ego, num_sensor


    @classmethod
    def from_file_multisweep_sample_data(cls,
                                         x_center,
                                         y_center,
                                         nusc: 'NuScenes',
                                         ref_sd_rec: Dict,
                                         return_trans_matrix: bool = False,
                                         flag_init: bool = False,
                                         min_distance: float = 1.0):
        """
        Added by Yiming. 2020/9/21
        Upperbound dataloader: transform the sweeps into the local coordinate of agent 0, 
        then translating to a fixed point (x_center, y_center)
        :param x_center, y_center: centre location of multiple lidar points
        :param ref_sd_rec: The current sample data record (lidar_top_id_0)
        :param return_trans_matrix: Whether need to return the transformation matrix
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        current_sd_rec = ref_sd_rec
        trans_matrix_list = list()
        skip_frame = 0
        sample_record = nusc.get('sample', ref_sd_rec['sample_token'])
        num_sensor = int(len(sample_record['data'])/2)
        x_ego = [[] for i in range(num_sensor)]
        y_ego = [[] for i in range(num_sensor)]

        x_filter = [[] for i in range(num_sensor)]
        y_filter = [[] for i in range(num_sensor)]

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])

            y_ego[k] = current_pose_rec['translation'][1]
            x_ego[k] = current_pose_rec['translation'][0]

            x_filter[k] = x_ego[k] - x_ego[0]
            y_filter[k] = y_ego[k] - y_ego[0]

        if flag_init == False:
           x_center = sum(x_filter)/num_sensor
           y_center = sum(y_filter)/num_sensor

        translation_vector = [-x_center, -y_center, 0]

        for k in range(num_sensor):
            # Load up the pointcloud.
            pointsensor_token = sample_record['data']['LIDAR_TOP_id_' + str(k)]
            current_sd_rec = nusc.get('sample_data', pointsensor_token)
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            
            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(np.sum([current_pose_rec['translation'],translation_vector],axis=0),
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

            current_pc.transform(trans_matrix)

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * (current_sd_rec['timestamp']+k)  # positive difference

            if k % (skip_frame + 1) == 0:
                times = time_lag * np.ones((1, current_pc.nbr_points()))
            else:
                times = time_lag * np.ones((1, 1))  # dummy value
            all_times = np.hstack((all_times, times))

            all_pc.points = np.hstack((all_pc.points, current_pc.points))

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        if return_trans_matrix:
            return all_pc, np.squeeze(all_times, 0), trans_matrix_list, x_center, y_center, num_sensor, x_ego, y_ego
        else:
            return all_pc, np.squeeze(all_times, 0), x_center, y_center, num_sensor, x_ego, y_ego


    @classmethod
    def get_instance_boxes_upperbound_sample_data(cls,
                                                  x_center,
                                                  y_center,
                                                  x_ego,
                                                  y_ego,
                                                  nusc: 'NuScenes',
                                                  ref_sd_rec: Dict,
                                                  instance_token: str,
                                                  nsweeps_back: int = 5,
                                                  nsweeps_forward: int = 5) -> \
            Tuple[List['Box'], np.array, List[str], List[str]]:
        """
        Added by Yiming. 2020/9/21
        translate the boxes to a fixed point (x_center, y_center)
        :param nusc: A NuScenes instance.
        :param ref_sd_rec: The current sample data record.
        :param instance_token: The current selected instance.
        :param nsweeps_back: Number of sweeps to aggregate. The sweeps trace back.
        :param nsweeps_forward: Number of sweeps to aggregate. The sweeps are obtained from the future.
        :return: (list of bounding boxes, the time stamps of bounding boxes, attribute list, category list)
        """

        # Init
        box_list = list()
        all_times = list()
        attr_list = list()  # attribute list
        cat_list = list()  # category list

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Get the bounding boxes across different sweeps
        boxes = list()

        # Move backward to get the past annotations
        current_sd_rec = ref_sd_rec
        for _ in range(nsweeps_back):
            box, attr, cat = nusc.get_instance_box(current_sd_rec['token'], instance_token)
            boxes.append(box)  # It is possible the returned box is None
            attr_list.append(attr)
            cat_list.append(cat)

            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
            all_times.append(time_lag)

            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        # Move forward to get the future annotations
        current_sd_rec = ref_sd_rec

        # Abort if there are no future sweeps.
        if current_sd_rec['next'] != '':
            current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

            for _ in range(nsweeps_forward):
                box, attr, cat = nusc.get_instance_box(current_sd_rec['token'], instance_token)
                boxes.append(box)  # It is possible the returned box is None
                attr_list.append(attr)
                cat_list.append(cat)

                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # negative difference
                all_times.append(time_lag)

                if current_sd_rec['next'] == '':
                    break
                else:
                    current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        percep_radius = 32
        translation_vector = [x_center, y_center, 0]

        # Map the bounding boxes to the local sensor coordinate
        for box in boxes:
            save_flag = False
            if box is not None:
                for k in range(len(x_ego)):
                    if np.max(np.abs(np.array([box.center[0],box.center[1]])-np.array([x_ego[k],y_ego[k]]))) < percep_radius:
                       # Move box to ego vehicle coord system
                       box.translate(-np.array(ref_pose_rec['translation'])-np.array(translation_vector))
                       box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

                       # Move box to sensor coord system
                       box.translate(-np.array(ref_cs_rec['translation']))
                       box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)

                       # caused by coordinate inconsistency of nuscene-toolkit
                       box.center[0] = - box.center[0]
#                       box_to_ego = transform_matrix(np.sum([ref_pose_rec['translation'],translation_vector],axis=0),
#                                               Quaternion(ref_pose_rec['rotation']), inverse=True)
#                       ego_to_sensor = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
#                       
                       save_flag = True
                       break

                if save_flag == False:
                   box.translate(-np.array([100000,100000,100000]))

            box_list.append(box)
        return box_list, all_times, attr_list, cat_list


    @classmethod
    def get_instance_boxes_multisweep_sample_data(cls,
                                                  nusc: 'NuScenes',
                                                  ref_sd_rec: Dict,
                                                  instance_token: str,
                                                  nsweeps_back: int = 5,
                                                  nsweeps_forward: int = 5) -> \
            Tuple[List['Box'], np.array, List[str], List[str]]:
        """
        Return the bounding boxes associated with the given instance. The bounding boxes are across different sweeps.
        For each bounding box, we need to map its (global) coordinates to the reference frame.
        For this function, the reference sweep is supposed to be from sample data record (not sample. ie, keyframe).
        :param nusc: A NuScenes instance.
        :param ref_sd_rec: The current sample data record.
        :param instance_token: The current selected instance.
        :param nsweeps_back: Number of sweeps to aggregate. The sweeps trace back.
        :param nsweeps_forward: Number of sweeps to aggregate. The sweeps are obtained from the future.
        :return: (list of bounding boxes, the time stamps of bounding boxes, attribute list, category list)
        """

        # Init
        box_list = list()
        all_times = list()
        attr_list = list()  # attribute list
        cat_list = list()  # category list

        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Get the bounding boxes across different sweeps
        boxes = list()

        # Move backward to get the past annotations
        current_sd_rec = ref_sd_rec
        for _ in range(nsweeps_back):
            box, attr, cat = nusc.get_instance_box(current_sd_rec['token'], instance_token)
            boxes.append(box)  # It is possible the returned box is None
            attr_list.append(attr)
            cat_list.append(cat)

            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
            all_times.append(time_lag)

            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        # Move forward to get the future annotations
        current_sd_rec = ref_sd_rec

        # Abort if there are no future sweeps.
        if current_sd_rec['next'] != '':
            current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

            for _ in range(nsweeps_forward):
                box, attr, cat = nusc.get_instance_box(current_sd_rec['token'], instance_token)
                boxes.append(box)  # It is possible the returned box is None
                attr_list.append(attr)
                cat_list.append(cat)

                time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # negative difference
                all_times.append(time_lag)

                if current_sd_rec['next'] == '':
                    break
                else:
                    current_sd_rec = nusc.get('sample_data', current_sd_rec['next'])

        # Map the bounding boxes to the local sensor coordinate
        for box in boxes:
            if box is not None:
                # Move box to ego vehicle coord system
                box.translate(-np.array(ref_pose_rec['translation']))
                box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

                # Move box to sensor coord system
                box.translate(-np.array(ref_cs_rec['translation']))
                box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)

                # caused by coordinate inconsistency of nuscene-toolkit
                box.center[0] = - box.center[0]

            box_list.append(box)
        #print(temp)
        return box_list, all_times, attr_list, cat_list

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(self,
                      ax: Axes,
                      view: np.ndarray = np.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax: Axes,
                         view: np.ndarray = np.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self,
                       color_channel: int,
                       ax: Axes,
                       view: np.ndarray,
                       x_lim: Tuple[float, float],
                       y_lim: Tuple[float, float],
                       marker_size: float) -> None:
        """
        Helper function for rendering.
        :param color_channel: Point channel to use as color.
        :param ax: Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])


class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)


class RadarPointCloud(PointCloud):

    # Class-level settings for radar pointclouds, see from_file().
    invalid_states = [0]  # type: List[int]
    dynprop_states = range(7)  # type: List[int] # Use [0, 2, 6] for moving objects only.
    ambig_states = [3]  # type: List[int]

    @classmethod
    def disable_filters(cls) -> None:
        """
        Disable all radar filter settings.
        Use this method to plot all radar returns.
        Note that this method affects the global settings.
        """
        cls.invalid_states = list(range(18))
        cls.dynprop_states = list(range(8))
        cls.ambig_states = list(range(5))

    @classmethod
    def default_filters(cls) -> None:
        """
        Set the defaults for all radar filter settings.
        Note that this method affects the global settings.
        """
        cls.invalid_states = [0]
        cls.dynprop_states = range(7)
        cls.ambig_states = [3]

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 18

    @classmethod
    def from_file(cls,
                  file_name: str,
                  invalid_states: List[int] = None,
                  dynprop_states: List[int] = None,
                  ambig_states: List[int] = None) -> 'RadarPointCloud':
        """
        Loads RADAR data from a Point Cloud Data file. See details below.
        :param file_name: The path of the pointcloud file.
        :param invalid_states: Radar states to be kept. See details below.
        :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
        :param ambig_states: Radar states to be kept. See details below.
        To keep all radar returns, set each state filter to range(18).
        :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.

        Example of the header fields:
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
        TYPE F F F I I F F F F F I I I I I I I I
        COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        WIDTH 125
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS 125
        DATA binary

        Below some of the fields are explained in more detail:

        x is front, y is left

        vx, vy are the velocities in m/s.
        vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
        We recommend using the compensated velocities.

        invalid_state: state of Cluster validity state.
        (Invalid states)
        0x01	invalid due to low RCS
        0x02	invalid due to near-field artefact
        0x03	invalid far range cluster because not confirmed in near range
        0x05	reserved
        0x06	invalid cluster due to high mirror probability
        0x07	Invalid cluster because outside sensor field of view
        0x0d	reserved
        0x0e	invalid cluster because it is a harmonics
        (Valid states)
        0x00	valid
        0x04	valid cluster with low RCS
        0x08	valid cluster with azimuth correction due to elevation
        0x09	valid cluster with high child probability
        0x0a	valid cluster with high probability of being a 50 deg artefact
        0x0b	valid cluster but no local maximum
        0x0c	valid cluster with high artefact probability
        0x0f	valid cluster with above 95m in near range
        0x10	valid cluster with high multi-target probability
        0x11	valid cluster with suspicious angle

        dynProp: Dynamic property of cluster to indicate if is moving or not.
        0: moving
        1: stationary
        2: oncoming
        3: stationary candidate
        4: unknown
        5: crossing stationary
        6: crossing moving
        7: stopped

        ambig_state: State of Doppler (radial velocity) ambiguity solution.
        0: invalid
        1: ambiguous
        2: staggered ramp
        3: unambiguous
        4: stationary candidates

        pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
        0: invalid
        1: <25%
        2: 50%
        3: 75%
        4: 90%
        5: 99%
        6: 99.9%
        7: <=100%
        """

        assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

        meta = []
        with open(file_name, 'rb') as f:
            for line in f:
                line = line.strip().decode('utf-8')
                meta.append(line)
                if line.startswith('DATA'):
                    break

            data_binary = f.read()

        # Get the header rows and check if they appear as expected.
        assert meta[0].startswith('#'), 'First line must be comment'
        assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
        sizes = meta[3].split(' ')[1:]
        types = meta[4].split(' ')[1:]
        counts = meta[5].split(' ')[1:]
        width = int(meta[6].split(' ')[1])
        height = int(meta[7].split(' ')[1])
        data = meta[10].split(' ')[1]
        feature_count = len(types)
        assert width > 0
        assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
        assert height == 1, 'Error: height != 0 not supported!'
        assert data == 'binary'

        # Lookup table for how to decode the binaries.
        unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                         'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                         'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
        types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

        # Decode each point.
        offset = 0
        point_count = width
        points = []
        for i in range(point_count):
            point = []
            for p in range(feature_count):
                start_p = offset
                end_p = start_p + int(sizes[p])
                assert end_p < len(data_binary)
                point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
                point.append(point_p)
                offset = end_p
            points.append(point)

        # A NaN in the first point indicates an empty pointcloud.
        point = np.array(points[0])
        if np.any(np.isnan(point)):
            return cls(np.zeros((feature_count, 0)))

        # Convert to numpy matrix.
        points = np.array(points).transpose()

        # If no parameters are provided, use default settings.
        invalid_states = cls.invalid_states if invalid_states is None else invalid_states
        dynprop_states = cls.dynprop_states if dynprop_states is None else dynprop_states
        ambig_states = cls.ambig_states if ambig_states is None else ambig_states

        # Filter points with an invalid state.
        valid = [p in invalid_states for p in points[-4, :]]
        points = points[:, valid]

        # Filter by dynProp.
        valid = [p in dynprop_states for p in points[3, :]]
        points = points[:, valid]

        # Filter by ambig_state.
        valid = [p in ambig_states for p in points[11, :]]
        points = points[:, valid]

        return cls(points)


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

    def copy(self) -> 'Box':
        """
        Create a copy of self.
        :return: A copy.
        """
        return copy.deepcopy(self)
