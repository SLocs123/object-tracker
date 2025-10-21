from boxmot.motion.kalman_filters.Traj_KF.Utils.simple_kf import SimpleKalmanFilterXY, SimpleKalmanFilterWH
from boxmot.motion.kalman_filters.Traj_KF.Utils.multi_kf import MultiKalman
from boxmot.motion.kalman_filters.Traj_KF.Utils.transformations import create_traj_map
from boxmot.motion.kalman_filters.Traj_KF.Utils.utils import read_traj, is_within
from boxmot.motion.kalman_filters.Traj_KF.Utils.Occlusion_detect import OcclusionDetect
import numpy as np


from boxmot.motion.kalman_filters.Traj_KF.Utils.metabus import bus

class Trajectory_Filter():
    def __init__(self, traj_dir):
        """
        Initializes the Traj_KF class.

        Args:
            traj_dir (str): The directory path to the trajectory data in pickle format.
            KF_Type (str): The type of Kalman Filter to use. Possible values are 'traj' or 'image'.

        Attributes:
            traj (NoneType): Placeholder for trajectory data.
            polygon_set (dict): A dictionary containing polygon data loaded from the pickle file.
            polygons (list): A list of polygons extracted from the polygon_set.
            assigned (NoneType): Placeholder for assigned data.
            trajectories (NoneType): Placeholder for trajectory information.
            sr (NoneType): Placeholder for spatial reference or related data.
        """

        self.polygon_set = read_traj(traj_dir)
        self.polygons = self.polygon_set.pop('polygons')

        self.simple_kf_xy = SimpleKalmanFilterXY()
        self.multi_kf_xy = MultiKalman()
        self.kf_box = SimpleKalmanFilterWH()
        self.occldet = OcclusionDetect()

        self.define_traj_sr_maps()
        
    
    def initiate(self, track):
        """
        Always starts in image domain.
        """
        self.simple_kf_xy.initiate(track)
        self.kf_box.initiate(track)

        track.last_updated = 0
        
        track.history = []
        track.mean = self.combine_mean(track.xymean, track.whmean)
        return track.mean, [track.xycov, track.whcov]
        
    def predict(self, track):
        """
        Run Kalman filter prediction step. look into handling multi----might just pass the track to handle below!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -------------------------------- need to check if track is assigned, handle different domains
        """

        if track.assigned:
            self.multi_kf_xy.predict(track)
        else:   
            self.simple_kf_xy.predict(track)
        
        self.kf_box.predict(track)
            
        track.mean = self.combine_mean(track.xymean, track.whmean)
        track.last_updated+=1

        return track.mean, [track.xycov, track.whcov]
        
    def update(self, track, xywh):
        """
        Update the states of the trajectory tracker, assign and correct tracks and create necassary maps
        Might need to update xywh everytime, need to look at typicaly track syntax
        """
        xy = xywh[:2]
        wh = xywh[2:4]
        
        
        occlusion, history, info = self.occldet.step(track.mean[:4], track.history, xywh)
        track.history = history
        
        if occlusion:
            """update assuming constant motion model, need to implement a new motion model for occlusion"""
            bus.put_track(track.id, "occluded", True)
            bus.put_track(track.id, "info", info)
        else:
            bus.put_track(track.id, "occluded", False)
            bus.put_track(track.id, "info", info)


        if not track.assigned:
            track.assigned = is_within(xy, self.polygons)

            self.simple_kf_xy.update(track, xy)
            self.kf_box.update(track, wh)

            if track.assigned and track.assigned in self.all_maps:
                track.maps = self.all_maps[track.assigned]
                track.xywh = [*xy, *wh]
                self.multi_kf_xy.initiate(track)
            else:
                track.assigned = None  # Revert invalid assignment

        else:
            self.multi_kf_xy.update(track, xy, wh)
            self.kf_box.update(track, wh)
            flag = "assigned"

            
        track.mean = self.combine_mean(track.xymean, track.whmean)
        track.last_updated = 0 

        return track.mean, [track.xycov, track.whcov] # this might lead to duplicate code for some trackers, but should function normally
        
    
    def define_traj_sr_maps(self):
        self.all_maps = {}
        for ext_key, internal_dict in self.polygon_set.items():
            trajectories = []
            for trajs in internal_dict.values():
                trajectories.append(np.array(trajs[:, 0]))
            self.all_maps[ext_key] = create_traj_map(trajectories)
        
    def split_mean(self, mean):
        """
        Splits the mean into xy and wh components.
        For mean = [x, y, w, h, vx, vy, vw, vh]:
          xy = [x, y, vx, vy]
          wh = [w, h, vw, vh]
        """
        xy = [mean[0], mean[1], mean[4], mean[5]]
        wh = [mean[2], mean[3], mean[6], mean[7]]
        return xy, wh
    
    def combine_mean(self, xymean, whmean):
        """
        Combines xy and ah components into a single mean vector.
        """
        return np.array([xymean[0], xymean[1], whmean[0], whmean[1], xymean[2], xymean[3], whmean[2], whmean[3]])
        
