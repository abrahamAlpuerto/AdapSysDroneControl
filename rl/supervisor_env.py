import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.failure_aviary import FailureAviary
from ctrl.str_controller import STRController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class SupervisorEnv(gym.Env):
    """
    RL Environment for training a supervisor to tune STR adaptation.
    The action is the RLS forgetting factor lambda.
    """
    
    def __init__(self, 
                 gui=False, 
                 failure_time_range=(2.0, 5.0),
                 episode_len_sec=10.0):
        super(SupervisorEnv, self).__init__()
        
        self.gui = gui
        self.failure_time_range = failure_time_range
        self.episode_len_sec = episode_len_sec
        self.ctrl_freq = 240
        self.max_steps = int(self.episode_len_sec * self.ctrl_freq)
        
        # Action space: lambda in [0.9, 0.999]
        # We'll map RL output [-1, 1] to [0.9, 0.999]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space (12 dimensions):
        # 0-2: pos_error (target - current)
        # 3-5: vel
        # 6:   rls_residual
        # 7-10: motor_effectiveness_estimates
        # 11:  current_lambda
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.env = None
        self.ctrl = None
        self.steps = 0
        self.failure_time = 0
        self.failed_motor_idx = 0
        self.current_lambda = 0.99

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.env is not None:
            self.env.close()
            
        # Randomize failure
        self.failure_time = np.random.uniform(*self.failure_time_range)
        self.failed_motor_idx = np.random.randint(0, 4)
        
        # Re-instantiate environment and controller
        self.env = FailureAviary(gui=self.gui, num_drones=1, ctrl_freq=self.ctrl_freq)
        self.ctrl = STRController(drone_model=DroneModel.CF2X, lambda_factor=0.99)
        
        self.steps = 0
        self.current_lambda = 0.99
        
        obs, info = self.env.reset()
        return self._get_obs(obs[0]), {}

    def _get_obs(self, aviary_obs):
        # aviary_obs: 0:3 pos, 3:7 quat, 7:10 rpy, 10:13 vel, 13:16 ang_v, 16:20 actions, 20:24 mask, 24: z_accel
        pos_error = self.env.target_pos - aviary_obs[0:3]
        vel = aviary_obs[10:13]
        residual = self.ctrl.rls.get_residual()
        eff = self.ctrl.get_effectiveness()
        
        return np.hstack([
            pos_error,
            vel,
            residual,
            eff,
            self.current_lambda
        ]).astype(np.float32)

    def step(self, action):
        # Map action [-1, 1] to [0.9, 0.999]
        self.current_lambda = 0.9 + (action[0] + 1) / 2 * 0.099
        self.ctrl.rls.lam = self.current_lambda
        
        t = self.steps / self.ctrl_freq
        
        # Current state from aviary
        # Note: We need the last observation to get accel_z for the controller
        # But computeControl also updates RLS.
        # To avoid double update, we'll manually get state and call computeControl.
        
        # We need the state from the internal env
        aviary_obs = self.env._computeObs()[0]
        
        # Trigger failure
        if t >= self.failure_time and np.all(self.env.failure_mask[0] == 1.0):
            self.env.fail_motor(0, self.failed_motor_idx)
            
        # Compute Control
        action_rpms, pos_e, yaw_e = self.ctrl.computeControl(
            control_timestep=1.0/self.ctrl_freq,
            cur_pos=aviary_obs[0:3],
            cur_quat=aviary_obs[3:7],
            cur_vel=aviary_obs[10:13],
            cur_ang_vel=aviary_obs[13:16],
            target_pos=self.env.target_pos,
            observed_accel_z=aviary_obs[24],
            mass=self.env.M
        )
        
        # Step inner environment
        next_aviary_obs, _, terminated, truncated, info = self.env.step(action_rpms.reshape(1, 4))
        
        self.steps += 1
        
        # Calculate Reward
        # Penalty for position error
        dist = np.linalg.norm(self.env.target_pos - next_aviary_obs[0, 0:3])
        # Penalty for RLS residual (high residual means poor estimation)
        resid = abs(self.ctrl.rls.get_residual())
        
        reward = -1.0 * dist - 0.1 * resid
        
        # Survival bonus if healthy
        if not terminated:
            reward += 0.1
            
        # Termination conditions
        done = False
        if next_aviary_obs[0, 2] < 0.2: # Crashed
            reward -= 10.0
            done = True
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(next_aviary_obs[0]), reward, done, False, info

    def render(self):
        pass

    def close(self):
        if self.env:
            self.env.close()
