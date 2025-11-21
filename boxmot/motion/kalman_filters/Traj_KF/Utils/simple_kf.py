import numpy as np
import scipy.linalg


# from .xy_noise_depth import Kf_noise
# from .xy_noise_half import Kf_noise
# from .xy_noise_basic import Kf_noise
from .wh_noise_basic_small_process import Kf_noise

class SimpleKalmanFilterXY:
    def __init__(self, dim=2, std_pos=1. / 20, std_vel=1. / 160):
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
        self._std_weight_position = std_pos
        self._std_weight_velocity = std_vel
        self._min_std = 1e-3

    def _dynamic_std(self, covariance):
        diag = np.diag(covariance)
        pos_scale = np.sqrt(np.maximum(diag[:self.dim], self._min_std))
        vel_scale = np.sqrt(np.maximum(diag[self.dim:], self._min_std))
        return pos_scale, vel_scale

    def initiate(self, track):
        xy = track.xywh[:2]
        mean_pos = xy
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        scale = np.mean(xy)  # rough proxy for noise scaling
        std = [
            self._std_weight_position * scale for _ in range(self.dim)
        ] + [
            self._std_weight_velocity * scale for _ in range(self.dim)
        ]

        covariance = np.diag(np.square(std))
        track.xymean = mean
        track.xycov = covariance

    def predict(self, track):
        mean = track.xymean
        covariance = track.xycov

        pos_scale, vel_scale = self._dynamic_std(covariance)
        std = np.r_[
            self._std_weight_position * pos_scale,
            self._std_weight_velocity * vel_scale,
        ]

        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        track.xymean = mean
        track.xycov = covariance

    def update(self, track, measurement):
        mean = track.xymean
        covariance = track.xycov

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T

        pos_scale, _ = self._dynamic_std(covariance)
        std = np.maximum(
            self._std_weight_position * pos_scale,
            self._min_std,
        )
        innovation_cov = np.diag(np.square(std))
        projected_cov += innovation_cov

        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), (covariance @ self._update_mat.T).T
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        track.xymean = new_mean
        track.xycov = new_cov


class SimpleKalmanFilterWH:
    def __init__(self, dim=2, vanishing_point_y: float | None = None, image_height: float | None = None):
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
        self._min_std = 1e-3

        self.noise_model = Kf_noise()
        self.noise_model.image_height = image_height
        self.noise_model.vanishing_point = vanishing_point_y

    def initiate(self, track):
        wh = track.xywh[2:4]
        y = track.xywh[1]+ wh[1]/2 # use bottom of box for depth
        mean_pos = wh
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]


        std = self.noise_model._get_initial_covariance_std(wh, box_y=y)
        covariance = np.diag(np.square(std))


        track.whmean = mean
        track.whcov = covariance

    def predict(self, track):
        mean = track.whmean
        covariance = track.whcov

        std_pos, std_vel = self.noise_model._get_process_noise_std(mean[:2], box_y=track.xywh[1]+ (mean[1]/2)) # use bottom of box for depth

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean

        # print("mean shape:", mean.shape)
        # print("xymean: ", mean)
        # print("cov shape:", covariance.shape)
        # print("motion mat shape:", self._motion_mat.shape)
        # print("motion cov shape:", motion_cov.shape)
        # print("motion cov:", motion_cov)
        # print("----")
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        track.whmean = mean
        track.whcov = covariance

    def update(self, track, measurement, xy):
        """this needs to be fixed, review the means being used, SHould i be returning whole or partial mean, where is it handled?
        Where are the inputs? handle whole track or not? probably whole track
        """
        mean = track.whmean
        covariance = track.whcov
        
        std = self.noise_model._get_measurement_noise_std(measurement[:2], box_y=xy[1]+(measurement[1]/2))
        projected_mean, projected_cov = self.project(mean, covariance, std)

        try: 
            chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            print("Projected cov:", projected_cov)
            print("Mean:", mean)
            print("std:", std)
            print("box_y: ", (track.xywh[1]+mean[1]))
            print("Measurement:", measurement, xy)
            std = self.noise_model._get_measurement_noise_std(measurement[:2], box_y=xy[1]+(measurement[1]/2), debug=True)
            raise   

        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        track.whmean = new_mean
        track.whcov = new_covariance
    
    def project(self, mean, covariance, std):
        """This should be fine, shouldnt be interacting with externals"""
        std = np.asarray(std, dtype=float)
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov


