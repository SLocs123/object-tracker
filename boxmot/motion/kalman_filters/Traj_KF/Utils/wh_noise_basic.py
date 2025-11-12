from typing import Tuple
import numpy as np

class box_noise():
    """Noise helpers for an XY-only constant-velocity Kalman filter.
    State order assumed: [x, y, vx, vy].
    Access to (w, h) is used purely for scale-normalised noise.
    """

    def __init__(
        self,
        _std_weight_position = 1.0 / 20,
        _std_weight_velocity = 1.0 / 160
    ):
        self._std_weight_position = _std_weight_position
        self._std_weight_velocity = _std_weight_velocity

    # ---------- API expected by filter ----------

    def _get_initial_covariance_std(self, wh: list[float]) -> np.ndarray:
        
        return np.array([
            2*self._std_weight_position * wh[0],
            2*self._std_weight_position * wh[1],
            10*self._std_weight_velocity * wh[0],
            10*self._std_weight_velocity * wh[1],
        ])

    def _get_process_noise_std(self, wh: list[float]) -> tuple[np.ndarray, np.ndarray]:\
        
        return np.array([ # does this need to be reduced?????????????????? test
            self._std_weight_position * wh[0],
            self._std_weight_position * wh[1],
        ]), np.array([
            self._std_weight_velocity * wh[0],
            self._std_weight_velocity * wh[1],
        ])


    def _get_measurement_noise_std(self, wh: list[float]) -> np.ndarray:
        
        return np.array([
            self._std_weight_position * wh[0],
            self._std_weight_position * wh[1],
        ])
