import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.failure_aviary import FailureAviary
from ctrl.str_controller import STRController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class SupervisorEnv(gym.Env):
    """
    RL Environment for training a supervisor to tune STR adaptation.
    Actions: [lambda, P_gain_mult, I_gain_mult, D_gain_mult, Yaw_gain_mult]
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
        
        # Action space: [lambda, P_m, I_m, D_m, Yaw_m]
        # Range [-1, 1] mapped to:
        # lambda: [0.9, 0.999]
        # PID_m: [0.5, 2.0]
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
        # Observation space (18 dimensions):
        # 0-2: pos_error (target - current)
        # 3-5: vel
        # 6-9: quat (orientation)
        # 10-12: ang_vel
        # 13-16: motor_effectiveness_estimates
        # 17:  current_lambda
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.env = None
        self.ctrl = None
        self.steps = 0
        self.failure_time = 0
        self.failed_motor_idx = 0
        self.current_lambda = 0.99
        self.gain_multipliers = np.ones(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.env is not None:
            self.env.close()
            
        # Randomize failure
        self.failure_time = np.random.uniform(*self.failure_time_range)
        self.failed_motor_idx = np.random.randint(0, 4)
        
        # Re-instantiate environment and controller
        self.env = FailureAviary(gui=self.gui, num_drones=1, ctrl_freq=self.ctrl_freq)
        self.ctrl = STRController(drone_model=DroneModel.CF2X, 
                                 lambda_factor=0.99,
                                 kf=self.env.KF,
                                 km=self.env.KM,
                                 arm_length=self.env.L,
                                 mass=self.env.M,
                                 inertia=self.env.J)
        
        self.steps = 0
        self.current_lambda = 0.99
        self.gain_multipliers = np.ones(4)
        
        obs, info = self.env.reset()
        return self._get_obs(obs[0]), {}

    def _get_obs(self, aviary_obs):
        # aviary_obs (28 dims): 
        # 0:3 pos, 3:7 quat, 7:10 rpy, 10:13 vel, 13:16 ang_v, 16:20 actions, 20:24 mask, 24: z_accel, 25:28 ang_accel
        pos_error = self.env.target_pos - aviary_obs[0:3]
        vel = aviary_obs[10:13]
        quat = aviary_obs[3:7]
        ang_v = aviary_obs[13:16]
        eff = self.ctrl.get_effectiveness()
        
        return np.hstack([
            pos_error,
            vel,
            quat,
            ang_v,
            eff,
            self.current_lambda
        ]).astype(np.float32)

    def step(self, action):
        # Map actions
        # lambda: [-1, 1] -> [0.9, 0.999]
        self.current_lambda = 0.9 + (action[0] + 1) / 2 * 0.099
        self.ctrl.rls.lam = self.current_lambda
        
        # PID multipliers: [-1, 1] -> [0.5, 2.0]
        self.gain_multipliers = 0.5 + (action[1:] + 1) / 2 * 1.5
        self.ctrl.set_gain_multipliers(self.gain_multipliers)
        
        t = self.steps / self.ctrl_freq
        
        # We need the state from the internal env
        # Note: FailureAviary returns 28 dims in _computeObs
        aviary_obs = self.env._computeObs()[0]
        
        # Trigger failure (Complete failure now)
        if t >= self.failure_time and np.all(self.env.failure_mask[0] == 1.0):
            self.env.fail_motor(0, self.failed_motor_idx, failed_power=0.0)
            
        # Compute Control
        action_rpms, pos_e, yaw_e = self.ctrl.computeControl(
            control_timestep=1.0/self.ctrl_freq,
            cur_pos=aviary_obs[0:3],
            cur_quat=aviary_obs[3:7],
            cur_vel=aviary_obs[10:13],
            cur_ang_vel=aviary_obs[13:16],
            target_pos=self.env.target_pos,
            observed_accel_z=aviary_obs[24],
            observed_ang_accel=aviary_obs[25:28],
            mass=self.env.M
        )
        
        # Step inner environment
        next_aviary_obs, _, terminated, truncated, info = self.env.step(action_rpms.reshape(1, 4))
        
        self.steps += 1
        
        # Calculate Reward
        cur_state = next_aviary_obs[0]
        pos_error = np.linalg.norm(self.env.target_pos - cur_state[0:3])
        
        # Orientation error: penalty for not being level (upright is [0,0,0,1] or similar)
        # Simplest: penalty for roll and pitch (cur_state[7:9])
        rpy = cur_state[7:10]
        att_error = np.linalg.norm(rpy[0:2])
        
        # Control effort penalty
        effort = np.mean(action_rpms) / self.env.MAX_RPM
        
        reward = -1.0 * pos_error - 0.5 * att_error - 0.1 * effort
        
        # Survival bonus
        if not terminated:
            reward += 0.5
            
        # Termination conditions
        done = False
        if cur_state[2] < 0.15: # Crashed
            reward -= 20.0
            done = True
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(cur_state), reward, done, False, info

    def render(self):
        pass

    def close(self):
        if self.env:
            self.env.close()
