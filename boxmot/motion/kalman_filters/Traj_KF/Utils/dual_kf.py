from .simple_kf import SimpleKalmanFilterXY, SimpleKalmanFilterWH
        
class DualKalman:
    def __init__(self, dim1=2, dim2=2):
        self.kf_traj = SimpleKalmanFilter(dim=2)  # for x/y in trajectory domain
        self.kf_box = SimpleKalmanFilter(dim=2)   # for a/h in image domain

    def initiate(self, traj_measurement=None, box_measurement=None):
        self.traj_mean, self.traj_cov = self.kf_traj.initiate(traj_measurement)
        self.box_mean, self.box_cov = self.kf_box.initiate(box_measurement)
    
    def reset(self, traj_reset=None, box_reset=None):
        """
        Reset the Kalman filter states. This is used to reinitialise the filter when switching domains, typically only traj_reset is used.
        Args:
            traj_reset (tuple): Initial state for trajectory Kalman filter.
            box_reset (tuple): Initial state for box Kalman filter.
        """
        if traj_reset is not None:
            self.traj_mean, self.traj_cov = self.kf_traj.initiate(traj_reset)
        if box_reset is not None:
            self.box_mean, self.box_cov = self.kf_box.initiate(box_reset)

    def predict(self):
        self.traj_mean, self.traj_cov = self.kf_traj.predict(self.traj_mean, self.traj_cov)
        self.box_mean, self.box_cov = self.kf_box.predict(self.box_mean, self.box_cov)
        return self.traj_mean, self.box_mean

    def update(self, traj_measurement, box_measurement):
        self.traj_mean, self.traj_cov = self.kf_traj.update(self.traj_mean, self.traj_cov, traj_measurement)
        self.box_mean, self.box_cov = self.kf_box.update(self.box_mean, self.box_cov, box_measurement)

    def get_state(self):
        return {
            'trajectory': (self.traj_mean, self.traj_cov),
            'box': (self.box_mean, self.box_cov)
        }
        