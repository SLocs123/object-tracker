from .Utils.simple_kf import SimpleKalmanFilterXY, SimpleKalmanFilterWH
from .Utils.multi_kf import MultiKalman
from .Utils.transformations import create_traj_map
from .Utils.utils import read_traj, is_within
# from .Utils.Occlusion_detect import OcclusionDetect
from .Utils.Occlusion_detect_static import OcclusionDetect
import numpy as np


from .Utils.metabus import bus

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
        # self.image_meta = self.polygon_set.pop('image_meta') !!!!-------------------- need to implement into setup
        self.image_meta = {'width':3840, 'height':2160, 'vanishing_point_y':344}  # temporary fix
        

        self.simple_kf_xy = SimpleKalmanFilterXY()
        self.multi_kf_xy = MultiKalman(vanishing_point_y=self.image_meta['vanishing_point_y'], image_height=self.image_meta['height'])
        self.kf_box = SimpleKalmanFilterWH(vanishing_point_y=self.image_meta['vanishing_point_y'], image_height=self.image_meta['height'])
        self.occldet = OcclusionDetect()

        self.define_traj_sr_maps()
        
    
    def initiate(self, track):
        """
        Always starts in image domain.
        """
        # print("running")
        self.simple_kf_xy.initiate(track)
        self.kf_box.initiate(track)

        track.last_updated = 0
        
        track.history = []
        track.occluded = False
        track.mean = self.combine_mean(track.xymean, track.whmean)
        return track.mean, [track.xycov, track.whcov]
        
    def predict(self, track):
        """
        Run Kalman filter prediction step. look into handling multi----might just pass the track to handle below!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -------------------------------- need to check if track is assigned, handle different domains
        """
        # occlusion, history, info = self.occldet.step(track.mean[:4], track.history, xywh)
        # track.history = history
        
        if track.assigned:
            if track.occluded:
                avg_delta, last_wh = self.get_clean_delta(track.history)
                # print(f'traj_kf, avg_delta: {avg_delta}, last_wh: {last_wh}')
                self.multi_kf_xy.predict_occluded(track, avg_delta)
                track.whmean[:2] = last_wh  # keep size from last clean detection
            else:
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
        track.occluded = occlusion
        
        if occlusion:
            """update assuming constant motion model, need to implement a new motion model for occlusion"""
            bus.put_track(track.id, "occluded", True)
            bus.put_track(track.id, "info", info)
        # else:
            # bus.put_track(track.id, "occluded", False)
            # bus.put_track(track.id, "info", info)
        if not track.assigned:
            track.assigned = is_within(xy, self.polygons)

            self.simple_kf_xy.update(track, xy)
            self.kf_box.update(track, wh, xy)

            if track.assigned and track.assigned in self.all_maps:
                track.maps = self.all_maps[track.assigned]
                track.xywh = [*xy, *wh]
                self.multi_kf_xy.initiate(track,)
            else:
                track.assigned = None  # Revert invalid assignment

        else:
            self.multi_kf_xy.update(track, xy, wh)
            self.kf_box.update(track, wh, xy)

            
        track.mean = self.combine_mean(track.xymean, track.whmean)
        track.last_updated = 0 

        return track.mean, [track.xycov, track.whcov] # this might lead to duplicate code for some trackers, but should function normally
        
    def get_clean_delta(self, history):
        """
        Get clean delta from the multi-kalman filter.
        """
                  
        # Safely filter only entries whose phase equals the string 'clean'.
        # Handle both Python str and numpy scalar string types.
        def is_clean_phase(item):
            phase = item.get('phase', None)
            if isinstance(phase, str):
                return phase == 'clean'
            # numpy scalar string (e.g. np.array('clean') or np.str_) supports .item()
            try:
                return phase.item() == 'clean'
            except Exception:
                return False
            
            
        last_clean = []
        for h in history:
            if is_clean_phase(h):
                last_clean.append(np.asarray(h['xywh'], dtype=float))
            else:
                break
        
        try:
            last_wh = np.asarray(last_clean[-1][2:4], dtype=float)
        except IndexError:
            last_wh = np.array([50,50]) # default to a square if no clean history
                
        delta = []
        last_xy = None

        for xywh in last_clean:
            if last_xy is not None:
                delta.append(xywh[:2] - last_xy)
            last_xy = xywh[:2]
            
        if len(delta) == 0:
            avg_delta = 0
        else:
            avg_axis = np.mean(np.asarray(delta), axis=0)
            avg_delta = np.linalg.norm(avg_axis)
            
            
        # print('---------------- GET CLEAN DELTA ----------------')
        # print(f'Histroy: {history}')
        # print(f'History length: {len(history)}')
        # print(f'Last clean entries: {len(last_clean)}')
        # print(f'Avg delta: {avg_delta}, Last wh: {last_wh}')
        return avg_delta, last_wh
    
    def define_traj_sr_maps(self):
        self.all_maps = {}
        for ext_key, internal_dict in self.polygon_set.items():
            trajectories = []
            for data_dict in internal_dict.values():
                trajs = data_dict['trajectory']
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
        
