import numpy as np

class RLSEstimator:
    """Recursive Least Squares (RLS) estimator for drone motor effectiveness."""

    def __init__(self, num_motors=4, lambda_factor=0.99, p_init=1000.0, theta_init=None):
        """
        Initialize the RLS estimator.

        Parameters
        ----------
        num_motors : int
            Number of motors (default: 4).
        lambda_factor : float
            Forgetting factor (0 < lambda <= 1). Typical values: 0.95 to 1.0.
        p_init : float
            Initial value for the diagonal elements of the covariance matrix P.
        theta_init : float or ndarray, optional
            Initial estimate for the thrust coefficients. If None, uses a typical KF value.
        """
        self.n = num_motors
        self.lam = lambda_factor
        
        # Initial estimate theta (thrust coefficients kf)
        if theta_init is None:
            # Typical KF is around 3.16e-10 for Crazyflie 2.x
            self.theta = np.ones(self.n) * 3.16e-10
        elif isinstance(theta_init, (int, float)):
            self.theta = np.ones(self.n) * theta_init
        else:
            self.theta = np.array(theta_init)
            
        # Initial covariance matrix P
        self.P = np.eye(self.n) * p_init
        
        # Residual tracking
        self.last_residual = 0.0

    def update(self, observed_accel_z, rpms, mass, gravity=9.81):
        """
        Update the parameter estimates based on new observations.

        Model: a_z = (1/m) * sum(k_i * RPM_i^2) - g
        Re-arranged for RLS: y = a_z + g = sum(k_i * (RPM_i^2 / m))
        y = phi^T * theta
        phi = [RPM_1^2/m, RPM_2^2/m, RPM_3^2/m, RPM_4^2/m]^T
        theta = [k1, k2, k3, k4]^T

        Parameters
        ----------
        observed_accel_z : float
            Measured vertical acceleration in the world frame (or body frame if aligned).
        rpms : ndarray
            Current RPMs of the motors.
        mass : float
            Mass of the drone.
        gravity : float
            Acceleration due to gravity.
        """
        # Measurement y
        y = observed_accel_z + gravity
        
        # Regressor phi (n, 1)
        phi = (rpms**2 / mass).reshape(-1, 1)
        
        # Innovation / Error
        prediction = (phi.T @ self.theta)[0]
        epsilon = y - prediction
        self.last_residual = epsilon
        
        # Gain vector K (n, 1)
        # K = P * phi / (lambda + phi^T * P * phi)
        denominator = self.lam + (phi.T @ self.P @ phi)[0, 0]
        K = (self.P @ phi) / denominator
        
        # Update theta
        self.theta = self.theta + (K * epsilon).flatten()
        
        # Update P
        # P = (1/lambda) * (P - K * phi^T * P)
        self.P = (1.0 / self.lam) * (self.P - K @ phi.T @ self.P)
        
        return self.theta

    def get_estimates(self):
        """Returns the current estimated thrust coefficients."""
        return self.theta

    def get_residual(self):
        """Returns the last measurement residual."""
        return self.last_residual
