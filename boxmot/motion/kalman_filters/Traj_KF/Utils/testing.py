import numpy as np

def _depth_factor(
    self,
    y: float,
    vanishing_point_y: float | None = None,
    image_height: float | None = None,
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

    if vanishing_point_y is None or image_height is None:
        return round(fallback, 2)

    # Simple linear interpolation between min and max
    depth_factor = np.clip(
        (y - vanishing_point_y) / (image_height - vanishing_point_y),
        min_factor,
        max_factor,
    )
    return round(depth_factor, 2)

def _apply_depth_factor(
    self,
    noise: list[float],
    depth_factor: float,
    gamma: float = 1.0,
    w_p: float = 0.5,
    w_m: float = 1.0,
    noise_type: str = "process"
) -> float:
    """Apply depth factor to noise scaling."""
    S = (1.0 - depth_factor) ** gamma

    if noise_type == "process":
        return noise * (S**w_p)
    elif noise_type == "measurement":
        return noise * (S**w_m)
    else:
        raise ValueError(f"Invalid noise_type '{noise_type}', must be 'process' or 'measurement'")
    
    
    
    
y = 30000
vanishing_point_y = 344
image_height = 2160
factor = _depth_factor(None, y, vanishing_point_y, image_height)
print(f"Depth factor at y={y}: {factor}")