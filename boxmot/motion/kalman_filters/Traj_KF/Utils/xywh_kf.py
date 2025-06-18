from typing import Tuple

import numpy as np

from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter


class noise_from_wh():
    """
    A Kalman filter for tracking bounding boxes in image space with state space:
        x, y, w, h, vx, vy, vw, vh
    """

    def __init__(self):
        self._std_weight_position = 1. / 20  # Standard deviation weight for position noise
        self._std_weight_velocity = 1. / 80  # Standard deviation weight for velocity noise


    def _get_initial_covariance_std(self, wh: np.ndarray) -> np.ndarray:
        # wh: [w, h]
        return np.array([
            2 * self._std_weight_position * wh[0],
            2 * self._std_weight_position * wh[1],
            10 * self._std_weight_velocity * wh[0],
            10 * self._std_weight_velocity * wh[1],
        ])

    def _get_process_noise_std(self, wh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # wh: [w, h]
        std_pos = [
            self._std_weight_position * wh[0],
            self._std_weight_position * wh[1],
        ]
        
        std_vel = [
            self._std_weight_velocity * wh[0],
            self._std_weight_velocity * wh[1],
        ]
        
        return np.array(std_pos), np.array(std_vel)
    
    # def get_process_noise_from_cov_and_time(self,
    #                                         cov: np.ndarray,
    #                                         time_since_update: int,
    #                                         base_pos_scale: float = 1.0,
    #                                         base_vel_scale: float = 1.0,
    #                                         pos_diffusion: float = 0.3,
    #                                         vel_diffusion: float = 0.1,
    #                                         min_std: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Generate linearly increasing process noise stds from current covariance, with separate diffusion
    #     rates and scaling for position and velocity components.

    #     Args:
    #         cov (np.ndarray): Current 4x4 covariance matrix (for [x, y, vx, vy]).
    #         time_since_update (int): Number of frames since last update.
    #         base_pos_scale (float): Base multiplier for position stds.
    #         base_vel_scale (float): Base multiplier for velocity stds.
    #         pos_diffusion (float): Linear growth rate per frame for position noise.
    #         vel_diffusion (float): Linear growth rate per frame for velocity noise.
    #         min_std (float): Lower bound for all stds to ensure numerical stability.

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: (std_pos, std_vel)
    #     """

    #     diag = np.diag(cov)
    #     pos_std = np.sqrt(np.maximum(diag[:2], min_std))
    #     vel_std = np.sqrt(np.maximum(diag[2:4], min_std))

    #     t = max(0, time_since_update - 1)

    #     # Apply base scaling and linear diffusion separately
    #     std_pos = np.maximum(pos_std * (base_pos_scale + t * pos_diffusion), min_std)
    #     std_vel = np.maximum(vel_std * (base_vel_scale + t * vel_diffusion), min_std)

    #     return std_pos, std_vel

    # def _get_measurement_noise_std(self, wh: np.ndarray) -> np.ndarray:
    #     area = max(wh[0] * wh[1], 1e-2)
    #     inv_area = 1.0 / area
    #     scale = np.sqrt(inv_area)
    #     std_noise = [
    #         self._std_weight_position * scale,
    #         self._std_weight_position * scale,
    #     ]
        
    #     return np.array(std_noise)
    
    # def _get_measurement_noise_std(self, wh: np.ndarray) -> np.ndarray:
    #     w, h = np.maximum(wh, 1.0)
    #     scale = min(w, h) + 0.5 * abs(w - h)
    #     std = self._std_weight_position * scale
    #     return np.array([std, std])
    

    def _get_measurement_noise_std(self, mean: np.ndarray):
        std_noise = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
        ]
        return std_noise

    def _get_multi_process_noise_std(self, wh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # wh: shape (N, 2), columns are [w, h]
        std_pos = [
            self._std_weight_position * wh[:, 0],
            self._std_weight_position * wh[:, 1],
        ]
        
        std_vel = [
            self._std_weight_velocity * wh[:, 0],
            self._std_weight_velocity * wh[:, 1],
        ]
        
        return np.array(std_pos), np.array(std_vel)
