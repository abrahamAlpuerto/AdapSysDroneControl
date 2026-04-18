import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from ctrl.rls_estimator import RLSEstimator

class STRController(DSLPIDControl):
    """Indirect Self-Tuning Regulator (STR) for quadrotors."""

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 lambda_factor: float=0.998,
                 min_effectiveness: float=0.0,
                 kf=3.16e-10,
                 km=7.94e-12,
                 arm_length=0.0397,
                 mass=0.027,
                 inertia=np.array([1.4e-5, 1.4e-5, 2.2e-5])
                 ):
        """
        Initialize the STR controller.
        """
        super().__init__(drone_model=drone_model, g=g)
        
        # Store gravity
        self.G = g
        
        # Initialize RLS estimator with 4D measurement
        self.rls = RLSEstimator(num_motors=4, 
                                lambda_factor=lambda_factor,
                                kf=kf,
                                km=km,
                                arm_length=arm_length,
                                mass=mass,
                                inertia=inertia)
        
        self.min_eff = min_effectiveness
        self.last_rpms = np.zeros(4)
        
        # RL-tuned parameters (multipliers for PID gains)
        # Default to 1.0 (no change)
        self.gain_multipliers = np.ones(4) # [P, I, D, R] (Proportional, Integral, Derivative, Roll/Pitch priority)
        
        # Precompute nominal B matrix (Control Effectiveness Matrix)
        self.B_nom = self._compute_B(np.ones(4))
        self.B_nom_inv = np.linalg.pinv(self.B_nom)
        
        # Warmup counter to allow filters and dynamics to settle
        self.step_counter = 0
        self.warmup_steps = 100

    def _compute_B(self, effectiveness):
        """
        Computes the control effectiveness matrix B(k) such that
        [Thrust, Roll_m, Pitch_m, Yaw_m]^T = B(k) * RPM^2
        Matching DSLPIDControl circular indexing.
        """
        kf = self.rls.kf
        km = self.rls.km
        l_const = self.rls.l_const
        k = effectiveness**2
        
        B = np.zeros((4, 4))
        # 0:FR (+x, -y, CCW), 1:RR (-x, -y, CW), 2:RL (-x, +y, CCW), 3:FL (+x, +y, CW)
        
        # Thrust: Sum of all thrusts
        B[0, :] = k * kf
        
        # Roll Moment (L-R)
        B[1, 0] = - k[0] * l_const * kf
        B[1, 1] = - k[1] * l_const * kf
        B[1, 2] = + k[2] * l_const * kf
        B[1, 3] = + k[3] * l_const * kf
        
        # Pitch Moment (Rear-Front)
        B[2, 0] = - k[0] * l_const * kf
        B[2, 1] = + k[1] * l_const * kf
        B[2, 2] = + k[2] * l_const * kf
        B[2, 3] = - k[3] * l_const * kf
        
        # Yaw Moment (CW-CCW)
        B[3, 0] = - k[0] * km
        B[3, 1] = + k[1] * km
        B[3, 2] = - k[2] * km
        B[3, 3] = + k[3] * km
        
        return B

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
                       observed_ang_accel=None,
                       mass=None,
                       manual_k=None
                       ):
        """
        Computes the adaptive control action using a Pseudo-Inverse Mixer.
        """
        # 1. Update RLS (only after warmup)
        if observed_accel_z is not None and observed_ang_accel is not None:
            if self.step_counter > self.warmup_steps:
                self.rls.update(observed_accel_z, observed_ang_accel, self.last_rpms, self.G)
            self.step_counter += 1
            
        # 2. Get nominal PID output (RPMs)
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
        
        # 3. Extract desired Thrust and Moments from nominal RPMs
        desired_efforts = self.B_nom @ (nominal_rpms**2)
        
        # 4. Apply adaptive mixer
        k_hat = manual_k if manual_k is not None else self.rls.get_estimates()
        k_hat = np.clip(k_hat, self.min_eff, 1.2)
        
        min_k = np.min(k_hat)

        # Skip auto-damping as we need more control authority
        # # Auto-Dampen PID efforts if failure is severe
        # if min_k < 0.9:
        #     # Reduce Roll/Pitch efforts to prevent wild oscillations in 3-motor flight
        #     desired_efforts[1:3] *= 0.6
        #     # Zero out Yaw command: independent yaw control is impossible with 3 motors
        #     # Letting it spin freely prevents actuator saturation.
        #     desired_efforts[3] = 0.0
        #     # Increase Thrust slightly to compensate for lost motor
        #     desired_efforts[0] *= 1.1
            
        # Apply RL-tuned multipliers on top of auto-dampening
        desired_efforts[0] *= self.gain_multipliers[0]
        desired_efforts[1:3] *= self.gain_multipliers[1]
        desired_efforts[3] *= self.gain_multipliers[3]
        
        B_hat = self._compute_B(k_hat)
        
        # Weighted Control Allocation
        # W is the output weight (how much we care about each error)
        yaw_weight = 1 if min_k > 0.1 else 0.001 
        W = np.diag([1, 1.2, 1.2, 1]) 
        W_half = np.sqrt(W)

        # Trying to weigh desired effort directly
        desired_efforts = W @ desired_efforts


        # -- This is a version without R --
        # Solve [W_half*B] * rpm2 = W_half*efforts
        new_rpm2 = np.linalg.pinv(W_half @ B_hat) @ (W_half @ desired_efforts)

        # -- This is a version with R --
        # We solve: min ||W_half(B*rpm2 - efforts)||^2 + ||R_half*rpm2||^2
        # # R is the input weight (cost of using each motor)
        # # We penalize using a motor proportional to its failure
        # # If k=1, cost is low. If k=0.1, cost is very high.
        # # r_i = 0.01 + (1.0 - k_i) * 10.0
        # r_coeffs = np.clip(0.01 + (1.0 - k_hat) * 10,1,120.0)
        # R_half = np.diag(np.sqrt(r_coeffs) * 1e-11) # Scale R to match B_hat's magnitude
        
        # # Solve using Augmented Least Squares:
        # # [W_half*B; R_half] * rpm2 = [W_half*efforts; 0]
        # A_aug = np.vstack([W_half @ B_hat, R_half])
        # b_aug = np.concatenate([W_half @ desired_efforts, np.zeros(4)])
        
        # new_rpm2 = np.linalg.pinv(A_aug, rcond=1e-8) @ b_aug

        # -- This is a version without W and R --
        # new_rpm2 = np.linalg.pinv(B_hat) @ desired_efforts
        
        new_rpm2 = np.clip(new_rpm2, 0, 40000**2)
        adapted_rpms = np.sqrt(new_rpm2)
        
        # 5. Low-pass filter the output RPMs to reduce oscillations
        # alpha_action = 0.9 provides a good balance between smoothing and lag
        alpha_action = 0.9
        smoothed_rpms = alpha_action * adapted_rpms + (1.0 - alpha_action) * self.last_rpms
        
        # Store for next RLS update (RLS should see the actual smoothed command)
        self.last_rpms = smoothed_rpms.copy()
        
        return smoothed_rpms, pos_e, yaw_e

    def get_rls_estimates(self):
        return self.rls.get_estimates()
    
    def get_effectiveness(self):
        return self.rls.get_estimates()
    
    def set_gain_multipliers(self, multipliers):
        """
        Set PID gain multipliers from RL agent.
        multipliers: [Thrust_m, Att_m, D_m, Yaw_m]
        """
        self.gain_multipliers = multipliers
