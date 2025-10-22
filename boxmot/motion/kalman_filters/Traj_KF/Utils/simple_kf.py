import numpy as np
import scipy.linalg


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
        wh = track.xywh[2:4]
        mean_pos = wh
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        scale = np.mean(wh)  # rough proxy for noise scaling
        std = [
            self._std_weight_position * scale for _ in range(self.dim)
        ] + [
            self._std_weight_velocity * scale for _ in range(self.dim)
        ]
        covariance = np.diag(np.square(std))
        track.whmean = mean
        track.whcov = covariance

    def predict(self, track):
        mean = track.whmean
        covariance = track.whcov

        pos_scale, vel_scale = self._dynamic_std(covariance)
        std = np.r_[
            self._std_weight_position * pos_scale,
            self._std_weight_velocity * vel_scale,
        ]
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        track.whmean = mean
        track.whcov = covariance

    def update(self, track, measurement):
        mean = track.whmean
        covariance = track.whcov

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

        track.whmean = new_mean
        track.whcov = new_cov


