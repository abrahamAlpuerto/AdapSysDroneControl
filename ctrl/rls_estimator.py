import numpy as np

class RLSEstimator:
    """Recursive Least Squares (RLS) estimator for drone motor effectiveness."""

    def __init__(self, 
                 num_motors=4, 
                 lambda_factor=0.99, 
                 p_init=1.0, 
                 theta_init=None,
                 kf=3.16e-10,
                 km=7.94e-12,
                 arm_length=0.099,
                 mass=0.027,
                 inertia=np.array([0.0023, 0.0023, 0.004])):
        """
        Initialize the RLS estimator.

        Parameters
        ----------
        num_motors : int
            Number of motors (default: 4).
        lambda_factor : float
            Forgetting factor (0 < lambda <= 1).
        p_init : float
            Initial covariance scale.
        theta_init : float or ndarray, optional
            Initial estimate for the effectiveness (default: 1.0).
        kf, km, arm_length, mass, inertia : float
            Physical parameters of the drone.
        """
        self.n = num_motors
        self.lam = lambda_factor
        
        # Initial estimate theta (effectiveness factors, 1.0 = healthy)
        if theta_init is None:
            self.theta = np.ones(self.n)
        elif isinstance(theta_init, (int, float)):
            self.theta = np.ones(self.n) * theta_init
        else:
            self.theta = np.array(theta_init)
            
        # Initial covariance matrix P (n x n)
        self.P = np.eye(self.n) * p_init
        
        # Physical constants
        self.kf = kf
        self.km = km
        self.L = arm_length
        self.M = mass
        # Ensure inertia is a 1D vector of diagonal elements [Ix, Iy, Iz]
        if isinstance(inertia, np.ndarray) and inertia.ndim == 2:
            self.I = np.diag(inertia)
        else:
            self.I = np.array(inertia)
        
        # Precompute constants for the regressor
        self.l_const = self.L * np.sqrt(2) / 2.0
        
        # Residual tracking
        self.last_residual = np.zeros(4)

    def update(self, observed_accel_z, observed_ang_accel, rpms, gravity=9.81):
        """
        Update the parameter estimates based on 4D observations.
        """
        # 0. Safety Check: If measurements are NaN or extreme, skip update
        if np.any(np.isnan(observed_accel_z)) or np.any(np.isnan(observed_ang_accel)) or \
           np.abs(observed_accel_z) > 100 or np.any(np.abs(observed_ang_accel) > 1000):
            return self.theta

        # Measurement y (4, 1)
        y = np.zeros((4, 1))
        y[0, 0] = observed_accel_z + gravity
        y[1:4, 0] = observed_ang_accel
        
        # Regressor Phi (4, n)
        phi = np.zeros((4, self.n))

        # Use a scaling factor for RPMs to keep Phi entries near order 1-100
        rpm_scale = 10000.0
        # Clip RPMs used for estimation to prevent explosion
        safe_rpms = np.clip(rpms, 0, 50000)
        rpm_norm2 = (safe_rpms / rpm_scale)**2
        kf_scaled = self.kf * (rpm_scale**2)
        km_scaled = self.km * (rpm_scale**2)
        
        # Phi entries matching DSLPIDControl / MIXER_MATRIX circular layout:
        # FR: 0, RR: 1, RL: 2, FL: 3
        phi[0, :] = (kf_scaled / self.M) * rpm_norm2

        phi[1, 0] = - (self.l_const * kf_scaled / self.I[0]) * rpm_norm2[0]
        phi[1, 1] = - (self.l_const * kf_scaled / self.I[0]) * rpm_norm2[1]
        phi[1, 2] = + (self.l_const * kf_scaled / self.I[0]) * rpm_norm2[2]
        phi[1, 3] = + (self.l_const * kf_scaled / self.I[0]) * rpm_norm2[3]

        phi[2, 0] = - (self.l_const * kf_scaled / self.I[1]) * rpm_norm2[0]
        phi[2, 1] = + (self.l_const * kf_scaled / self.I[1]) * rpm_norm2[1]
        phi[2, 2] = + (self.l_const * kf_scaled / self.I[1]) * rpm_norm2[2]
        phi[2, 3] = - (self.l_const * kf_scaled / self.I[1]) * rpm_norm2[3]
        
        phi[3, 0] = - (km_scaled / self.I[2]) * rpm_norm2[0]
        phi[3, 1] = + (km_scaled / self.I[2]) * rpm_norm2[1]
        phi[3, 2] = - (km_scaled / self.I[2]) * rpm_norm2[2]
        phi[3, 3] = + (km_scaled / self.I[2]) * rpm_norm2[3]
        
        # Innovation
        prediction = phi @ self.theta.reshape(-1, 1)
        epsilon = (y - prediction)*0.9
        self.last_residual = epsilon.flatten()
        
        # Weighted Update
        v_thrust = 0.005
        v_moment = 1
        V = np.diag([v_thrust, v_moment, v_moment, v_moment])
        
        try:
            # Gain K = P * Phi^T * inv(lambda * V + Phi * P * Phi^T)
            S = self.lam * V + phi @ self.P @ phi.T
            K = self.P @ phi.T @ np.linalg.pinv(S)
            
            # Update theta
            new_theta = self.theta + (K @ epsilon).flatten()
            
            # Update P
            new_P = (1.0 / self.lam) * (self.P - K @ phi @ self.P)
            
            # 1. Stability Check: If update produces NaNs, discard it and reset P
            if np.any(np.isnan(new_theta)) or np.any(np.isnan(new_P)):
                self.P = np.eye(self.n) * 1.0 # Reset covariance
                return self.theta
        
            self.theta = new_theta
            self.P = new_P
            
        except np.linalg.LinAlgError:
            # SVD/Inverse failure in RLS itself
            self.P = np.eye(self.n) * 1.0
            return self.theta
        
        # Regularization: small leak towards 1.2 for stability
        self.theta = 0.998 * self.theta + 0.002 * 1.2
        
        # Clip theta to [0, 1.2] 
        self.theta = np.clip(self.theta, 0.0, 1.2)
        
        return self.theta

    def get_estimates(self):
        """Returns the current estimated effectiveness factors."""
        return self.theta

    def get_residual(self):
        """Returns the norm of the last measurement residual."""
        return np.linalg.norm(self.last_residual)
