import numpy as np
import scipy.linalg

from boxmot.motion.kalman_filters.Traj_KF.Utils.transformations import (
    img_to_traj_domain,
    traj_to_img_domain,
)
from boxmot.motion.kalman_filters.Traj_KF.Utils.xy_noise import Kf_noise

noise = Kf_noise()


class MultiKalman:
    """Manages multiple Kalman filters for tracking trajectories."""

    def __init__(self, dim=2):
        """Initialise the multi-hypothesis trajectory filter."""
        self.kf = MultiKalmanFilterXY(dim)

    def initiate(self, track):
        """Initialise a Kalman filter per candidate trajectory map."""
        track.xymeans = []
        track.xycovs = []
        maps = track.maps
        std = noise._get_initial_covariance_std(track.xywh[2:4])

        if maps:
            for map_ in maps:
                xy = track.xywh[:2]
                track.long_lat = img_to_traj_domain(xy, map_)
                mean, cov = self.kf.initiate(track.long_lat, std)
                track.xymeans.append(mean)
                track.xycovs.append(cov)
        else:
            raise ValueError("track.maps is not defined. This must be initialised before using traj_kf.")

        self._update_active_hypothesis(track)

    def predict(self, track):
        """Predict the next state for each trajectory hypothesis."""
        means = track.xymeans
        covariances = track.xycovs
        time = track.last_updated

        wh = np.asarray(track.xywh[2:4], dtype=float)
        base_measurement_std = noise._get_measurement_noise_std(wh)

        for i, mean in enumerate(means):
            cov = covariances[i]
            std_pos, std_vel = noise.get_process_noise_from_cov_and_time(
                cov,
                time,
                base_measurement_std=base_measurement_std,
            )
            mean, cov = self.kf.predict(mean, cov, std_pos, std_vel)
            track.xymeans[i] = mean.copy()
            track.xycovs[i] = cov.copy()

        self._update_active_hypothesis(track)

    def update(self, track, measurement, wh):
        """Update every hypothesis with the incoming measurement."""
        measurement = np.asarray(measurement, dtype=float)
        wh = np.asarray(wh, dtype=float)
        base_std = noise._get_measurement_noise_std(wh)

        updated_means = []
        updated_covariances = []

        steps_since_update = track.last_updated + 1

        for i, mean in enumerate(track.xymeans):
            lat_long = img_to_traj_domain(measurement, track.maps[i])
            projected_cov = self.kf._update_mat @ track.xycovs[i] @ self.kf._update_mat.T
            innovation = lat_long - mean[:2]
            measurement_std = noise.adaptive_measurement_std(
                base_std,
                innovation,
                mean[2:4],
                projected_cov,
                steps_since_update,
            )
            mean, cov = self.kf.update(
                mean,
                track.xycovs[i],
                lat_long,
                measurement_std,
            )
            updated_means.append(mean)
            updated_covariances.append(cov)

        track.xymeans = updated_means
        track.xycovs = updated_covariances

        self._update_active_hypothesis(track)

    def _update_active_hypothesis(self, track):
        """Select the hypothesis with the lowest positional uncertainty."""
        if not getattr(track, "xymeans", None):
            return

        scores = [np.trace(cov[:2, :2]) for cov in track.xycovs]
        best_index = int(np.argmin(scores))

        best_mean = track.xymeans[best_index]
        best_cov = track.xycovs[best_index]
        point = traj_to_img_domain(best_mean[:2], track.maps[best_index])

        vxvy = best_mean[2:4].copy()
        track.xymean = np.array([point[0], point[1], vxvy[0], vxvy[1]], dtype=float)
        track.xycov = best_cov.copy()
        track.active_map_index = best_index


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

