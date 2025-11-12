from typing import Tuple

import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


class KalmanFilterXYWH(BaseKalmanFilter):
    """
    A Kalman filter for tracking bounding boxes in image space with state space:
        x, y, w, h, vx, vy, vw, vh
    """

    def __init__(self):
        super().__init__(ndim=4)

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> list:       
        return [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[list, list]:
        std_pos = [
            0.5*self._std_weight_position * mean[2],
            0.5*self._std_weight_position * mean[3],
            0.5*self._std_weight_position * mean[2],
            0.5*self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> list:
        std_noise = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        return std_noise
    
    def _get_multi_process_noise_std(self, mean: np.ndarray) -> Tuple[list,list]:
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        return std_pos, std_vel
