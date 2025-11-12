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
        self.vanishing_point = 344
        self.image_height = 2160

    def _get_box_size(self, wh: list[float]) -> float:
        """Compute box size for noise scaling."""
        # area = wh[0] * wh[1]
        # averaged_side = np.sqrt(area)
        # return averaged_side
        return (wh[0] + wh[1]) / 2.0
  
    def _depth_factor(
        self,
        y: float | None,
        min_factor: float = 0.3,
        max_factor: float = 1.0,
        default: str = "min"
    ) -> float:
        """Compute depth factor based on y-coordinate and vanishing point."""
        
        defaults = {
            "min": min_factor,
            "max": max_factor,
            "mid": (min_factor + max_factor) / 2,
        }
        if default not in defaults:
            raise ValueError(f"Invalid default '{default}', must be one of {list(defaults)}")

        # Fallback value if computation can't be done
        fallback = defaults[default]

        if self.vanishing_point is None or self.image_height is None or y is None:
            return round(fallback, 2)

        # Simple linear interpolation between min and max
        depth_factor = np.clip(
            (y - self.vanishing_point) / (self.image_height - self.vanishing_point),
            min_factor,
            max_factor,
        )
        return round(depth_factor, 2)

    def _apply_depth_factor(
        self,
        noise: float,
        depth_factor: float,
        gamma: float = 1.0,
        w_p: float = 0.5,
        w_m: float = 1.0,
        noise_type: str = "process"
    ) -> float:
        """Apply depth factor to noise scaling."""
        S = (1.0 - depth_factor) ** gamma

        if noise_type == "process":
            return float(noise * (S**w_p))
        elif noise_type == "measurement":
            return float(noise * (S**w_m))
        else:
            raise ValueError(f"Invalid noise_type '{noise_type}', must be 'process' or 'measurement'")
    
    def _get_initial_covariance_std(self, measurement: np.ndarray) -> list:
        xy = measurement[:2]
        wh = measurement[2:4]
        y_bottom = xy[1] + wh[1]
        size = self._get_box_size(wh) # type: ignore
        
        depth_factor = self._depth_factor(y_bottom)
        
        scale = self._apply_depth_factor(
            noise=size,
            depth_factor=depth_factor,
            noise_type="measurement"
            )
        scale = abs(scale)
        
        return [
            2 * self._std_weight_position * scale,
            2 * self._std_weight_position * scale,
            2 * self._std_weight_position * scale,
            2 * self._std_weight_position * scale,
            10 * self._std_weight_velocity * scale,
            10 * self._std_weight_velocity * scale,
            10 * self._std_weight_velocity * scale,
            10 * self._std_weight_velocity * scale,
        ]

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[list, list]:
        xy = mean[:2]
        wh = mean[2:4]
        y_bottom = xy[1] + wh[1]
        size = self._get_box_size(wh) # type: ignore
        
        depth_factor = self._depth_factor(y_bottom)
        
        scale = self._apply_depth_factor(
            noise=size,
            depth_factor=depth_factor,
            noise_type="process"
            )
        scale = abs(scale)
        
        
        std_pos = [
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
        ]
        std_vel = [
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
        ]
        return std_pos, std_vel

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> list:
        xy = mean[:2]
        wh = mean[2:4]
        y_bottom = xy[1] + wh[1]
        size = self._get_box_size(wh) # type: ignore
        
        depth_factor = self._depth_factor(y_bottom)
        
        scale = self._apply_depth_factor(
            noise=size,
            depth_factor=depth_factor,
            noise_type="measurement"
            )        
        scale = abs(scale)
        
        std_noise = [
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_position * scale,
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
