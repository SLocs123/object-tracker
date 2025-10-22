import numpy as np
from boxmot.motion.kalman_filters.Traj_KF.Utils.transformations import img_to_traj_domain, traj_to_img_domain
import scipy.linalg
from boxmot.motion.kalman_filters.Traj_KF.Utils.xy_noise import Kf_noise
import os

noise = Kf_noise()
class MultiKalman:
    """
    Manages multiple Kalman filters for tracking trajectories in a multi-dimensional space.

    Attributes:
        dim (int): Number of position dimensions (e.g., 2 for x, y or a, h).
        dt (float): Time step for the motion model. This can be dynamically adjusted for better tracking in the future.
        _motion_mat (np.ndarray): State transition matrix for the motion model.
        _update_mat (np.ndarray): Observation matrix for the measurement model.
        _std_weight_position (float): Standard deviation weight for position noise.
        _std_weight_velocity (float): Standard deviation weight for velocity noise.
        kf (SimpleKalmanFilter): Instance of a simple Kalman filter for state estimation.
    """

    def __init__(self, dim=2):
        """
        Initializes the MultiKalman object with the specified dimensions and noise parameters.

        :param dim: Number of position dimensions (2 → x, y).
        :type dim: int
        :param std_pos: Standard deviation weight for position noise. affects the weight of measurments
        :type std_pos: float
        :param std_vel: Standard deviation weight for velocity noise. affects the weight of the motion model
        :type std_vel: float
        """
        self.kf = MultiKalmanFilterXY(dim)

    def initiate(self, track):
        """
        Initializes trajectory Kalman filters for a set of measurements and corresponding maps saved in track.
        This function converts the provided measurements into the trajectory domain for each map,
        then initializes a Kalman filter for each resulting point. The means and covariances of
        the initialized filters are collected in the track.
        Args:
            track (object): An object containing trajectory data, including:
        Returns:
            tuple: A tuple containing two lists:
                - means: List of mean state vectors for each initialized Kalman filter.
                - covariances: List of covariance matrices for each initialized Kalman filter.
        Note:
            If `track.maps` is not provided, the function returns a warning.
        """

        track.xymeans = []
        track.xycovs = []
        maps = track.maps
        std = noise._get_initial_covariance_std(track.xywh[2:4])  # Extract width and height from xywh
        
        if maps:
            for map in maps:
                xy = track.xywh[:2] 

                track.long_lat = img_to_traj_domain(xy, map) 
                mean, cov = self.kf.initiate(track.long_lat, std)  
                track.xymeans.append(mean)
                track.xycovs.append(cov)
        else:
            raise ValueError("track.maps is not defined. This must be initialised before using traj_kf.")

    def predict(self, track):
        """
        Predicts the next state for multiple trajectories using the Kalman filter.

        Args:
            means (list or np.ndarray): List or array of mean state vectors for each trajectory.
            covariances (list or np.ndarray): List or array of covariance matrices corresponding to each mean.
            maps (list): List of map objects or transformation data for each trajectory.

        Returns:
            None

        Description:
            For each trajectory, this method applies the Kalman filter's predict step using the provided mean and covariance.
            The predicted trajectory state is then transformed to the image domain using the corresponding map.
        """
        
          # Extract width and height from xywh
        
        means = track.xymeans
        covariances = track.xycovs
        time = track.last_updated  # Time since last update, used for process noise calculation

        for i, mean in enumerate(means):
            cov = covariances[i]
            std_pos, std_vel = noise.get_process_noise_from_cov_and_time(cov, time)
            mean, cov = self.kf.predict(mean, cov, std_pos, std_vel)

            track.xymeans[i] = mean.copy()  # Ensure we copy the mean to avoid reference issues
            track.xycovs[i] = cov.copy()
        
        # Sort xymeans and xycovs by mean[1] in xymeans, maintaining their correspondence, and get the index
        sorted_index, sorted_pair = min(
            enumerate(zip(track.xymeans, track.xycovs)),
            key=lambda pair: abs(pair[1][0][1])
        )

        xymean, xycov = sorted_pair
        point = traj_to_img_domain(xymean[:2], track.maps[sorted_index])

        vxvy = xymean[2:4].copy()
        track.xymean = [point[0], point[1], vxvy[0], vxvy[1]]  # Update the mean with the predicted position in image domain


    def update(self, track, measurement, wh):
        """
        Updates the trajectory Kalman filters with new measurements.
        For each measurement in the input list, this method updates the corresponding
        Kalman filter state (mean and covariance) using the filter's update method.
        The updated means and covariances for all trajectories are returned as lists.
        Args:
            measurements (list or array-like): A list of measurement vectors, one for each trajectory.
        Returns:
            tuple: A tuple containing two lists:
                - updated_means (list): The updated state means for each trajectory.
                - updated_covariances (list): The updated state covariances for each trajectory.
        """
        
        confidence = getattr(track, "conf", getattr(track, "confidence", 0.0))
        wh = np.array(wh)  # Extract width and height from xywh
        std = noise._get_measurement_noise_std(wh)
        
    
        updated_means = []
        updated_covariances = []

        for i, mean in enumerate(track.xymeans):
            # Convert measurement to trajectory domain
            lat_long = img_to_traj_domain(measurement, track.maps[i])
            mean, cov = self.kf.update(mean, track.xycovs[i], lat_long, std, confidence)
            updated_means.append(mean)
            updated_covariances.append(cov)

        track.xymeans = updated_means
        track.xycovs = updated_covariances

    
class MultiKalmanFilterXY:
    def __init__(self, dim=2):
        """
        dim: Number of position dimensions (2 → x, y or a, h)
        std_pos: Standard deviation weight for position noise
        std_vel: Standard deviation weight for velocity noise
        """
        self.dim = dim
        self.dt = 1.

        self._motion_mat = np.eye(2 * dim)
        for i in range(dim):
            self._motion_mat[i, dim + i] = self.dt

        self._update_mat = np.eye(dim, 2 * dim)

    def initiate(self, lat_long, std):
            
        mean_pos = lat_long
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        covariance = np.diag(np.square(std))
        
        return mean, covariance

    def predict(self, mean, covariance, std_pos, std_vel):
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance
    
    def project(self, mean, covariance, std, confidence=0.0):
        """
        Project state distribution to measurement space.
        
        """

        # NSA Kalman algorithm from GIAOTracker, which proposes a formula to
        # adaptively calculate the noise covariance Rek:
        # Rk = (1 − ck) Rk
        # where Rk is the preset constant measurement noise covariance
        # and ck is the detection confidence score at state k. Intuitively,
        # the detection has a higher score ck when it has less noise,
        # which results in a low Re.
        std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement, std, confidence=0.0):
        
        projected_mean, projected_cov = self.project(mean, covariance, std, confidence)

        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)

        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        
        return new_mean, new_covariance