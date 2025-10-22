from typing import Optional, Tuple

import numpy as np


class Kf_noise:
    """Noise helpers for the trajectory-aware Kalman filters."""

    def __init__(self):
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 80
        self._min_std = 1e-3

    def _get_initial_covariance_std(self, wh: np.ndarray) -> np.ndarray:
        return np.array([
            2 * self._std_weight_position * wh[0],
            2 * self._std_weight_position * wh[1],
            10 * self._std_weight_velocity * wh[0],
            10 * self._std_weight_velocity * wh[1],
        ])

    def _get_process_noise_std(self, wh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * wh[0],
            self._std_weight_position * wh[1],
        ]

        std_vel = [
            self._std_weight_velocity * wh[0],
            self._std_weight_velocity * wh[1],
        ]

        return np.array(std_pos), np.array(std_vel)

    def get_process_noise_from_cov_and_time(
        self,
        cov: np.ndarray,
        time_since_update: int,
        base_measurement_std: Optional[np.ndarray] = None,
        min_std: float = 1e-2,
        process_base: float = 1.8,
        process_growth: float = 0.45,
        velocity_base: float = 1.2,
        velocity_growth: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        diag = np.diag(cov)
        pos_std_from_cov = np.sqrt(np.maximum(diag[:2], min_std))
        vel_std_from_cov = np.sqrt(np.maximum(diag[2:4], min_std))

        steps = max(1, int(time_since_update) + 1)

        if base_measurement_std is None:
            base_measurement_std = pos_std_from_cov
        else:
            base_measurement_std = np.asarray(base_measurement_std, dtype=float)

        base_measurement_std = np.maximum(base_measurement_std, self._min_std)

        process_ratio = process_base + process_growth * (steps - 1)
        std_pos = np.maximum(
            process_ratio * base_measurement_std,
            pos_std_from_cov,
        )

        velocity_ratio = velocity_base + velocity_growth * (steps - 1)
        std_vel = np.maximum(
            velocity_ratio * base_measurement_std,
            vel_std_from_cov,
        )

        return std_pos, std_vel

    def _get_measurement_noise_std(self, wh: np.ndarray) -> np.ndarray:
        area = max(wh[0] * wh[1], 1e-2)
        inv_area = 1.0 / area
        scale = np.sqrt(inv_area)
        std_noise = [
            self._std_weight_position * scale,
            self._std_weight_position * scale,
        ]

        return np.array(std_noise)

    def adaptive_measurement_std(
        self,
        base_std: np.ndarray,
        innovation: np.ndarray,
        velocity: np.ndarray,
        projected_cov: np.ndarray,
        steps_since_update: int,
        base_gain: float = 0.65,
        far_gain: float = 0.4,
        max_gain: float = 0.95,
        missed_for_max_gain: int = 10,
        innovation_threshold: float = 1.0,
        innovation_max: float = 3.0,
    ) -> np.ndarray:
        """Compute an adaptive measurement standard deviation.

        The innovation term fed to this function is the raw positional residual in the
        trajectory coordinate frame.  Internally we compare its magnitude against an
        "innovation reference" defined as ``base_std + max(|velocity|, base_std)``.
        With the default ``innovation_threshold`` of 1.0, residuals at roughly that
        scale are treated as routine noise and keep the update close to ``base_gain``
        (≈65% measurement influence).  Larger residuals are considered increasingly
        surprising: once the innovation is about three times the reference magnitude
        (``innovation_max`` of 3.0) the ratio saturates and the effective gain falls
        toward ``far_gain`` (≈40% influence at nominal recency, rising toward 90–95%
        as the track ages).  For example, if ``base_std`` is 0.2 in the trajectory
        units and the current velocity is 0.5, the reference magnitude becomes 0.7;
        a residual of ~0.7 keeps the default weighting, while ~2.1 or greater is
        flagged as highly surprising and down-weights the measurement accordingly.
        """
        # Ensure all inputs are numpy arrays and clamp the base standard deviation to
        # a small floor so downstream ratios remain numerically stable.
        base_std = np.maximum(np.asarray(base_std, dtype=float), self._min_std)
        innovation = np.asarray(innovation, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        projected_cov = np.asarray(projected_cov, dtype=float)

        # Convert the integer "steps since update" into a [0, 1] factor that rises as
        # detections are missed, enabling us to lean more on measurements over time.
        steps = max(1, int(steps_since_update))
        time_alpha = np.clip(
            (steps - 1) / max(1, missed_for_max_gain),
            0.0,
            1.0,
        )

        # Blend between the nominal and maximum Kalman gains, giving older tracks a
        # higher measurement influence and pulling distant hypotheses slightly closer.
        time_gain = base_gain + (max_gain - base_gain) * time_alpha
        far_gain_at_time = far_gain + (0.9 - far_gain) * time_alpha

        # Derive an expected innovation magnitude using the object's size and
        # velocity, so we only inflate measurement noise when the residual grows.
        innovation_reference = np.maximum(
            base_std,
            np.maximum(np.abs(velocity), self._min_std),
        ) + base_std
        innovation_ratio = np.abs(innovation) / np.maximum(innovation_reference, self._min_std)

        # Map the innovation ratio into [0, 1] to describe how surprising the
        # measurement is; larger surprises trigger more conservative updates.
        innovation_alpha = np.clip(
            (innovation_ratio - innovation_threshold)
            / max(1e-6, innovation_max - innovation_threshold),
            0.0,
            1.0,
        )

        # Reduce the target gain when the innovation is large so the filter trusts
        # the measurement less, while respecting the time-based lower bound.
        desired_gain = time_gain - (time_gain - far_gain_at_time) * innovation_alpha
        desired_gain = np.clip(desired_gain, far_gain_at_time, max_gain)

        # Translate the desired Kalman gain into measurement variance, ensuring it
        # never drops below the projected uncertainty so updates stay consistent.
        projected_var = np.maximum(np.diag(projected_cov), self._min_std ** 2)
        measurement_var = projected_var * (1.0 - desired_gain) / np.maximum(desired_gain, 1e-3)

        # Enforce a noise floor tied to the base standard deviation to prevent the
        # gain from exceeding our intended limits when the prediction is overconfident.
        base_var = np.square(base_std)
        measurement_var = np.maximum(measurement_var, base_var)

        # Return the adaptive measurement standard deviation, which the caller can
        # use to modulate the update step toward the desired weighting.
        return np.sqrt(measurement_var)

    def _get_multi_process_noise_std(self, wh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * wh[:, 0],
            self._std_weight_position * wh[:, 1],
        ]

        std_vel = [
            self._std_weight_velocity * wh[:, 0],
            self._std_weight_velocity * wh[:, 1],
        ]

        return np.array(std_pos), np.array(std_vel)

