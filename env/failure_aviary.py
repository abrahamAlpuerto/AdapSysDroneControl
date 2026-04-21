import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class FailureAviary(CtrlAviary):
    """Custom environment for simulating motor failure in quadrotors."""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of the FailureAviary.

        Parameters are the same as in CtrlAviary.
        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )
        
        # Initialize failure mask for each drone (1: healthy, 0: failed)
        # Shape: (num_drones, 4)
        self.failure_mask = np.ones((self.NUM_DRONES, 4), dtype=np.float32)
        
        # Target position for RL and control
        self.target_pos = np.array([0., 0., 1.0])
        
        # Track last velocity to compute acceleration
        self.last_vel = np.zeros((self.NUM_DRONES, 3))
        self.accel = np.zeros((self.NUM_DRONES, 3))
        self.last_ang_vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_accel = np.zeros((self.NUM_DRONES, 3))

    def fail_motor(self, drone_index: int, motor_index: int, failed_power: float=0.0):
        """Triggers a failure in a specific motor of a specific drone.

        Parameters
        ----------
        drone_index : int
            Index of the drone (0 to NUM_DRONES-1).
        motor_index : int
            Index of the motor (0 to 3).
        failed_power : float
            The power left available after the motor failed (0.0 to 1.0).
        """
        if 0 <= drone_index < self.NUM_DRONES and 0 <= motor_index < 4 and 0.0 <= failed_power <= 1.0:
            self.failure_mask[drone_index, motor_index] = failed_power
            print(f"[INFO] Motor {motor_index} of drone {drone_index} failed! ({failed_power*100:.2f}% power left)")
        else:
            print(f"[ERROR] Invalid drone, motor index, or failed power left: {drone_index}, {motor_index}, {failed_power}")

    def _preprocessAction(self, action):
        """Overrides _preprocessAction to apply the failure mask.

        Parameters
        ----------
        action : ndarray
            The commanded RPMs for each motor of each drone.

        Returns
        -------
        ndarray
            The clipped and masked RPMs.
        """
        # Clip action to [0, MAX_RPM]
        clipped_action = super()._preprocessAction(action)
        
        # Apply failure mask
        masked_action = clipped_action * self.failure_mask
        
        return masked_action

    def _observationSpace(self):
        """Overrides _observationSpace to include the failure mask, z-acceleration and angular accelerations.

        Returns
        -------
        spaces.Box
            The extended observation space (20 + 4 + 1 + 3 = 28 dimensions per drone).
        """
        obs_space = super()._observationSpace()
        low = obs_space.low
        high = obs_space.high
        
        # Append 4 dimensions for the failure mask (range [0, 1])
        # 1 dimension for z-acceleration
        # 3 dimensions for angular accelerations (roll, pitch, yaw)
        low_ext = np.array([[0., 0., 0., 0., -np.inf, -np.inf, -np.inf, -np.inf] for _ in range(self.NUM_DRONES)])
        high_ext = np.array([[1., 1., 1., 1., np.inf, np.inf, np.inf, np.inf] for _ in range(self.NUM_DRONES)])
        
        new_low = np.hstack([low, low_ext])
        new_high = np.hstack([high, high_ext])
        
        return spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    def _computeObs(self):
        """Overrides _computeObs to include the failure mask, z-acceleration and angular accelerations.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 28) with the extended state.
        """
        # Base observation (NUM_DRONES, 20)
        obs = super()._computeObs()
        
        # Update linear acceleration with LPF (Low pass filter)
        current_vel = obs[:, 10:13]
        new_accel = (current_vel - self.last_vel) / self.CTRL_TIMESTEP
        self.accel = 0.5 * new_accel + 0.5 * self.accel # LPF
        self.last_vel = current_vel.copy()
        
        # Convert Z-accel to body frame (simplified for vertical RLS)
        # # This is linearized to pointing directly up.
        # # Need to account for accelerations in other axes for a better estimate.
        # rpy = obs[:, 7:10]
        # cos_roll = np.cos(rpy[:, 0])
        # cos_pitch = np.cos(rpy[:, 1])
        # z_accel_body = (self.accel[:, 2] + self.G) / (cos_roll * cos_pitch + 1e-6) - self.G

        # Get z-accel in body frame with quanternion (should be more accurate than above)
        cur_quat = obs[:, 3:7]
        cur_rotation = np.zeros((np.shape(cur_quat)[0], 3))
        z_accel_body = np.zeros((np.shape(cur_quat)[0],1))
        for i in range(np.shape(cur_quat)[0]):
            cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat[i, :])).reshape(3, 3)
            z_accel_body[i] = np.dot(self.accel[i, :], cur_rotation[:, 2])
        



        
        # Update angular acceleration with LPF
        current_ang_vel = obs[:, 13:16]
        new_ang_accel = (current_ang_vel - self.last_ang_vel) / self.CTRL_TIMESTEP
        self.ang_accel = 0.2 * new_ang_accel + 0.8 * self.ang_accel # LPF to help with stability
        self.last_ang_vel = current_ang_vel.copy()
        
        # Concatenate with failure mask (NUM_DRONES, 4), z_accel (NUM_DRONES, 1) and ang_accel (NUM_DRONES, 3)
        extended_obs = np.hstack([obs, self.failure_mask, z_accel_body.reshape(-1, 1), self.ang_accel])
        
        return extended_obs

    def step(self,
             action
             ):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.GUI:
            if np.any(self.failure_mask != 1.0):
                # If failure happened, use different zoom
                cameraDistance = 0.7
            else:
                cameraDistance = 0.7
            p.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                                            cameraYaw=-30,
                                            cameraPitch=-30,
                                            cameraTargetPosition=self.pos[0],
                                            physicsClientId=self.CLIENT
                                                )
        

        
        return obs, reward, terminated, truncated, info