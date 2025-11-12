import numpy as np

class Kf_noise:
    """Noise helpers for an XY-only constant-velocity Kalman filter.
    State order assumed: [x, y, vx, vy].
    Access to (w, h) is used purely for scale-normalised noise.
    """

    def __init__(
        self,
        image_height: float | None = None,
        vanishing_point: float | None = None,
        _std_weight_position = 1.0 / 10,
        _std_weight_velocity = 1.0 / 40
    ):
        self._std_weight_position = _std_weight_position
        self._std_weight_velocity = _std_weight_velocity
        self.image_height = image_height
        self.vanishing_point = vanishing_point


    # ---------- helpers ----------
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

    # ---------- API expected by filter ----------

    def _get_initial_covariance_std(self, wh: list[float], box_y: float | None = None) -> np.ndarray:
        """Return initial covariance STDs for a new track in XY-only state."""
        box_size = self._get_box_size(wh)
        depth_factor = self._depth_factor(y=box_y,)
        
        scale = self._apply_depth_factor(
            noise=box_size,
            depth_factor=depth_factor,
            noise_type="measurement"
            )
            
        return np.array([
            self._std_weight_position * scale,
            self._std_weight_position * scale,
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
        ])

    def _get_process_noise_std(self, wh: list[float], box_y: float) -> tuple[np.ndarray, np.ndarray]:\
        
        box_size = self._get_box_size(wh)
        depth_factor = self._depth_factor(y=box_y,)
        
        scale = self._apply_depth_factor(
            noise=box_size,
            depth_factor=depth_factor,
            noise_type="process"
            )
        
        return np.array([ # does this need to be reduced?????????????????? test
            self._std_weight_position * scale,
            self._std_weight_position * scale,
        ]), np.array([
            self._std_weight_velocity * scale,
            self._std_weight_velocity * scale,
        ])


    def _get_measurement_noise_std(self, wh: list[float], box_y: float) -> np.ndarray:
        """Return measurement STDs for z = [x, y] (length-2)."""
        box_size = self._get_box_size(wh)
        depth_factor = self._depth_factor(y=box_y,)
        
        scale = self._apply_depth_factor(
            noise=box_size,
            depth_factor=depth_factor,
            noise_type="measurement"
            )
        
        return np.array([
            self._std_weight_position * scale,
            self._std_weight_position * scale,
        ])
