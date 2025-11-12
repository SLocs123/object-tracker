import numpy as np
import scipy.linalg

from .transformations import (
    img_to_traj_domain,
    traj_to_img_domain,
)
from .xy_noise_basic import Kf_noise
# from .xy_noise_depth import Kf_noise
# from.xy_noise_half import Kf_noise




class MultiKalman:
    """Manages multiple Kalman filters for tracking trajectories."""

    def __init__(self, vanishing_point_y: float | None = None, image_height: float | None = None, dim=2):
        """Initialise the multi-hypothesis trajectory filter."""
        self.kf = MultiKalmanFilterXY(dim)
        self.vanishing_point_y = vanishing_point_y
        self.image_height = image_height
        self.noise = Kf_noise(vanishing_point=vanishing_point_y, image_height=image_height)

    def initiate(self, track, ):
        """Initialise a Kalman filter per candidate trajectory map."""
        track.xymeans = []
        track.xycovs = []
        maps = track.maps
        
        wh = track.xywh[2:4]
        y = track.xywh[1]+ wh[1] # use bottom of box for depth
        std = self.noise._get_initial_covariance_std(wh, box_y=y)

        if maps:
            for map_ in maps:
                xy = track.xywh[:2]
                track.long_lat = img_to_traj_domain(xy, map_)
                mean, cov = self.kf.initiate(track.long_lat, std)
                track.xymeans.append(mean)
                track.xycovs.append(cov)
        else:
            raise ValueError("track.maps is not defined. This must be initialised before using traj_kf.")

    def predict(self, track):
        """Predict the next state for each trajectory hypothesis."""
        wh = track.mean[2:4]
        y = track.mean[1] + wh[1]  # use bottom of box for depth
        means = track.xymeans
        covariances = track.xycovs

        for i, mean in enumerate(means):
            cov = covariances[i]
            std_pos, std_vel = self.noise._get_process_noise_std(wh, box_y=y)
            mean, cov = self.kf.predict(mean, cov, std_pos, std_vel)
            track.xymeans[i] = mean.copy()
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
        """Update every hypothesis with the incoming measurement."""
        measurement = np.asarray(measurement, dtype=float)
        y = measurement[1] + wh[1]  # use bottom of box for depth
        wh = list(wh)

        updated_means = []
        updated_covariances = []

        for i, mean in enumerate(track.xymeans):
            lat_long = img_to_traj_domain(measurement, track.maps[i])
            std = self.noise._get_measurement_noise_std(wh, box_y=y)
            new_mean, new_cov = self.kf.update(mean, track.xycovs[i], lat_long, std)

            updated_means.append(new_mean)
            updated_covariances.append(new_cov)

        track.xymeans = updated_means
        track.xycovs = updated_covariances



class MultiKalmanFilterXY:
    def __init__(self, dim=2):
        """dim: Number of position dimensions (2 → x, y or a, h)"""
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

    def project(self, mean, covariance, std):
        """Project state distribution to measurement space."""
        std = np.asarray(std, dtype=float)
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, std):
        projected_mean, projected_cov = self.project(mean, covariance, std)

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

