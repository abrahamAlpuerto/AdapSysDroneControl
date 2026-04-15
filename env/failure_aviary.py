import numpy as np
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

    def fail_motor(self, drone_index: int, motor_index: int):
        """Triggers a failure in a specific motor of a specific drone.

        Parameters
        ----------
        drone_index : int
            Index of the drone (0 to NUM_DRONES-1).
        motor_index : int
            Index of the motor (0 to 3).
        """
        if 0 <= drone_index < self.NUM_DRONES and 0 <= motor_index < 4:
            self.failure_mask[drone_index, motor_index] = 0.0
            print(f"[INFO] Motor {motor_index} of drone {drone_index} failed!")
        else:
            print(f"[ERROR] Invalid drone or motor index: {drone_index}, {motor_index}")

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
        """Overrides _observationSpace to include the failure mask and z-acceleration.

        Returns
        -------
        spaces.Box
            The extended observation space (20 + 4 + 1 = 25 dimensions per drone).
        """
        obs_space = super()._observationSpace()
        low = obs_space.low
        high = obs_space.high
        
        # Append 4 dimensions for the failure mask (range [0, 1])
        # And 1 dimension for z-acceleration
        low_ext = np.array([[0., 0., 0., 0., -np.inf] for _ in range(self.NUM_DRONES)])
        high_ext = np.array([[1., 1., 1., 1., np.inf] for _ in range(self.NUM_DRONES)])
        
        new_low = np.hstack([low, low_ext])
        new_high = np.hstack([high, high_ext])
        
        return spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    def _computeObs(self):
        """Overrides _computeObs to include the failure mask and z-acceleration.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 25) with the extended state.
        """
        # Base observation (NUM_DRONES, 20)
        obs = super()._computeObs()
        
        # Update acceleration: (v_new - v_old) / dt
        # obs[:, 10:13] is velocity (VX, VY, VZ)
        current_vel = obs[:, 10:13]
        self.accel = (current_vel - self.last_vel) / self.CTRL_TIMESTEP
        self.last_vel = current_vel.copy()
        
        z_accel = self.accel[:, 2:3] # (NUM_DRONES, 1)
        
        # Concatenate with failure mask (NUM_DRONES, 4) and z_accel (NUM_DRONES, 1)
        extended_obs = np.hstack([obs, self.failure_mask, z_accel])
        
        return extended_obs
