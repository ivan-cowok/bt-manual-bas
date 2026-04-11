# Ball-specific Kalman filter with tuned Q (process noise) and R (measurement noise)
import numpy as np
import scipy.linalg


class BallKalmanFilter(object):
    """
    Kalman filter optimized for soccer ball tracking.
    
    Key differences from standard Kalman:
    - HIGHER process noise Q: Ball motion is unpredictable (kicks, bounces, direction changes)
    - LOWER measurement noise R: Ball detections are reliable
    
    This makes the filter trust observations more than predictions.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # BALL-SPECIFIC TUNING:
        # Process noise (Q): HIGHER than people (ball motion unpredictable)
        self._std_weight_position_process = 1. / 10    # Was 1/20 for people (2x higher)
        self._std_weight_velocity_process = 1. / 40    # Was 1/160 for people (4x higher)
        
        # Measurement noise (R): LOWER than people (detections are reliable)
        self._std_weight_position_measure = 1. / 40    # Was 1/20 for people (0.5x lower)

    def initiate(self, measurement):
        """Initialize new track."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initial covariance (unchanged)
        std = [
            2 * self._std_weight_position_process * measurement[2],
            2 * self._std_weight_position_process * measurement[3],
            2 * self._std_weight_position_process * measurement[2],
            2 * self._std_weight_position_process * measurement[3],
            10 * self._std_weight_velocity_process * measurement[2],
            10 * self._std_weight_velocity_process * measurement[3],
            10 * self._std_weight_velocity_process * measurement[2],
            10 * self._std_weight_velocity_process * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Kalman prediction step.
        
        Uses HIGHER process noise (Q) for unpredictable ball motion.
        """
        # Process noise (Q) - HIGHER for ball
        std_pos = [
            self._std_weight_position_process * mean[2],
            self._std_weight_position_process * mean[3],
            self._std_weight_position_process * mean[2],
            self._std_weight_position_process * mean[3]]
        std_vel = [
            self._std_weight_velocity_process * mean[2],
            self._std_weight_velocity_process * mean[3],
            self._std_weight_velocity_process * mean[2],
            self._std_weight_velocity_process * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project to measurement space.
        
        Uses LOWER measurement noise (R) for reliable ball detections.
        """
        # Measurement noise (R) - LOWER for ball (trust detections)
        std = [
            self._std_weight_position_measure * mean[2],
            self._std_weight_position_measure * mean[3],
            self._std_weight_position_measure * mean[2],
            self._std_weight_position_measure * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Vectorized prediction for multiple tracks."""
        std_pos = [
            self._std_weight_position_process * mean[:, 2],
            self._std_weight_position_process * mean[:, 3],
            self._std_weight_position_process * mean[:, 2],
            self._std_weight_position_process * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity_process * mean[:, 2],
            self._std_weight_velocity_process * mean[:, 3],
            self._std_weight_velocity_process * mean[:, 2],
            self._std_weight_velocity_process * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Kalman update step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state and measurements."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

