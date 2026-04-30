import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class PPOQuadEnv(gym.Env):
    def __init__(self, gui=False):
        super(PPOQuadEnv, self).__init__()
        self.gui = gui
        self.physicsClient = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        #4 motors, continuous values between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        #kinematics + RPMs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        self.thrust_coeff = 3.16e-10 
        self.base_rpm = 20000         
        self.max_rpm = 35000
        self.dt = 1.0 / 240.0
        self.target_pos = np.array([0.0, 0.0, 1.0])
        
        self.fault_enabled = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        




        if options and 'fault_enabled' in options:
            self.fault_enabled = options['fault_enabled']
            
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        
        startPos = [0, 0, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.quadId = p.loadURDF("../cf2x.urdf", startPos, startOrientation) 
        
        self.time_step = 0
        self.current_rpms = np.array([self.base_rpm] * 4, dtype=np.float32)
        
        return self._get_state(), {}

    def step(self, action):



        # Action of 1.0 = +5000 RPM, Action of -1.0 = -5000 RPM
        rpm_adjustments = action * 5000.0 
        self.current_rpms += rpm_adjustments
        self.current_rpms = np.clip(self.current_rpms, 0, self.max_rpm)
        



        forces = self.thrust_coeff * (self.current_rpms ** 2)
        
        t = self.time_step * self.dt
        if self.fault_enabled and t >= 4.0:
            forces[0] *= 0.5
            
        for i in range(4):
            p.applyExternalForce(self.quadId, i, 
                                 forceObj=[0, 0, forces[i]], 
                                 posObj=[0, 0, 0], 
                                 flags=p.LINK_FRAME)
            
        p.stepSimulation()
        self.time_step += 1
        
        next_state = self._get_state()
        reward, terminated = self._calculate_reward_and_done(next_state)
        



        return next_state, reward, terminated, False, {}

    def _get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.quadId)
        lin_vel, ang_vel = p.getBaseVelocity(self.quadId)
        euler = p.getEulerFromQuaternion(quat)
        
        state = np.array([
            pos[0], pos[1], pos[2],
            lin_vel[0], lin_vel[1], lin_vel[2],
            euler[0], euler[1], euler[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
            self.current_rpms[0] / self.max_rpm,
            self.current_rpms[1] / self.max_rpm,
            self.current_rpms[2] / self.max_rpm,
            self.current_rpms[3] / self.max_rpm
        ], dtype=np.float32)
        return state

    # def _calculate_reward_and_done(self, state):
    #     pos = state[0:3]
    #     vel = state[3:6]
    #     angles = state[6:9]
    #     ang_vel = state[9:12] # [roll_rate, pitch_rate, yaw_rate]
        
    #     dist_error = np.linalg.norm(self.target_pos - pos)
    #     vel_error = np.linalg.norm(vel)
    #     tilt_error = abs(angles[0]) + abs(angles[1])
        


    #     spin_error = np.linalg.norm(ang_vel[0:2])

    #     reward = 5.0 - (1.0 * dist_error) - (0.2 * vel_error) - (0.5 * tilt_error) - (0.5 * spin_error)
        
    #     if dist_error < 0.1 and vel_error < 0.2 and vel[2] > -0.5:
    #         reward += 5.0

    #     terminated = False
    #     if pos[2] < 0.2:  
    #         reward -= 2000
    #         terminated = True
    #     elif dist_error > 1.5: 
    #         reward -= 2000
    #         terminated = True
            
    #     return reward, terminated
    

    def _calculate_reward_and_done(self, state):
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        ang_vel = state[9:12] 
        
        dist_error = np.linalg.norm(self.target_pos - pos)
        vel_error = np.linalg.norm(vel)
        tilt_error = abs(angles[0]) + abs(angles[1])
        



        spin_error = np.linalg.norm(ang_vel[0:2])

        xy_error = np.linalg.norm(self.target_pos[0:2] - pos[0:2])
        z_error = abs(self.target_pos[2] - pos[2])

        reward = 5.0 - (1.0 * dist_error) - (0.2 * vel_error) - (0.5 * tilt_error) - (0.5 * spin_error)
        
        if dist_error < 0.1 and vel_error < 0.2 and vel[2] > -0.5:
            reward += 5.0

        # if dist_error > 1 or 1.2 < pos[2] < 0.5:
        #     reward -= 100

        reward -= (0.5 * xy_error)  
        reward -= (3.0 * z_error)   
        terminated = False
        if pos[2] < 0.2:  
            reward -= 2000
            terminated = True
        elif dist_error > 1.5: 
            reward -= 2000
            terminated = True
            
        return reward, terminated