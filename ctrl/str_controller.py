import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from ctrl.rls_estimator import RLSEstimator

class STRController(DSLPIDControl):
    """Indirect Self-Tuning Regulator (STR) for quadrotors."""

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 lambda_factor: float=0.99,
                 min_effectiveness: float=0.1
                 ):
        """
        Initialize the STR controller.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control.
        g : float, optional
            The gravitational acceleration.
        lambda_factor : float, optional
            Forgetting factor for the RLS estimator.
        min_effectiveness : float, optional
            Minimum effectiveness ratio to avoid division by zero/extremely high RPMs.
        """
        super().__init__(drone_model=drone_model, g=g)
        
        # Store gravity
        self.G = g
        
        # Initialize RLS estimator with nominal KF
        self.rls = RLSEstimator(num_motors=4, 
                                lambda_factor=lambda_factor, 
                                theta_init=self.KF)
        
        self.min_eff = min_effectiveness
        self.nominal_kf = self.KF
        self.last_rpms = np.zeros(4)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       observed_accel_z=None,
                       mass=None
                       ):
        """
        Computes the adaptive control action.

        First, updates the RLS estimator if observed_accel_z is provided.
        Then, computes the nominal PID control and scales it based on 
        estimated motor effectiveness.
        """
        
        # 1. Update RLS if we have new measurements
        if observed_accel_z is not None and mass is not None:
            self.rls.update(observed_accel_z, self.last_rpms, mass, self.G)
            
        # 2. Get nominal control action (RPMs)
        nominal_rpms, pos_e, yaw_e = super().computeControl(control_timestep=control_timestep,
                                                            cur_pos=cur_pos,
                                                            cur_quat=cur_quat,
                                                            cur_vel=cur_vel,
                                                            cur_ang_vel=cur_ang_vel,
                                                            target_pos=target_pos,
                                                            target_rpy=target_rpy,
                                                            target_vel=target_vel,
                                                            target_rpy_rates=target_rpy_rates
                                                            )
        
        # 3. Adapt the control action
        # Effectiveness ratio = estimated_kf / nominal_kf
        estimated_kfs = self.rls.get_estimates()
        eff_ratios = estimated_kfs / self.nominal_kf
        
        # Clip ratios to avoid extreme values
        eff_ratios = np.clip(eff_ratios, self.min_eff, 2.0)
        
        # Scale RPMs: commanded_rpm = nominal_rpm / sqrt(eff_ratio)
        adapted_rpms = nominal_rpms / np.sqrt(eff_ratios)
        
        # Store for next RLS update
        self.last_rpms = adapted_rpms.copy()
        
        return adapted_rpms, pos_e, yaw_e

    def get_rls_estimates(self):
        """Returns current RLS estimates of thrust coefficients."""
        return self.rls.get_estimates()
    
    def get_effectiveness(self):
        """Returns current effectiveness ratios (0 to 1)."""
        return self.rls.get_estimates() / self.nominal_kf
