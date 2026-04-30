import pybullet as p
import pybullet_data
import numpy as np
import math

class DQLQuadEnv:
    def __init__(self, gui=False):
        self.gui = gui
        self.physicsClient = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 3^4 = 81 discrete actions
        self.action_space_size = 81
        self.state_space_size = 16
        
        self.thrust_coeff = 3.16e-10  
        self.base_rpm = 20000        
        self.rpm_step = 100        
        self.max_rpm = 40000
        self.min_rpm = 0
        
        self.dt = 1.0 / 240.0
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.reset()

    def reset(self, fault_enabled=False):
        self.fault_enabled = fault_enabled 
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        
        startPos = [0, 0, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.quadId = p.loadURDF("../cf2x.urdf", startPos, startOrientation) 
        
        self.time_step = 0
        self.current_rpms = np.array([self.base_rpm] * 4, dtype=np.float32)
        
        return self._get_state()

    def _decode_action(self, action_int):
        adjustments = []
        temp = action_int
        for _ in range(4):
            val = temp % 3
            if val == 0:
                adjustments.append(0)
            elif val == 1:
                adjustments.append(self.rpm_step)
            elif val == 2:
                adjustments.append(-self.rpm_step)
            temp //= 3
        return np.array(adjustments)

    def step(self, action_int):
        rpm_adjustments = self._decode_action(action_int)
        self.current_rpms += rpm_adjustments
        self.current_rpms = np.clip(self.current_rpms, self.min_rpm, self.max_rpm)
        
        forces = self.thrust_coeff * (self.current_rpms ** 2)
        
        t = self.time_step * self.dt
        if t >= 4.0:
            forces[0] *= 0.5 
            

        for i in range(4):
            p.applyExternalForce(self.quadId, i, 
                                 forceObj=[0, 0, forces[i]], 
                                 posObj=[0, 0, 0], 
                                 flags=p.LINK_FRAME)
            
        p.stepSimulation()
        self.time_step += 1
        
        next_state = self._get_state()
        reward, done = self._calculate_reward_and_done(next_state)
        
        return next_state, reward, done

    def _get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.quadId)
        lin_vel, ang_vel = p.getBaseVelocity(self.quadId)
        euler = p.getEulerFromQuaternion(quat)
        
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r, rpm1, rpm2, rpm3, rpm4]
        state = np.array([
            pos[0], pos[1], pos[2],
            lin_vel[0], lin_vel[1], lin_vel[2],
            euler[0], euler[1], euler[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
            # NEW: Add normalized RPMs to the state space
            self.current_rpms[0] / self.max_rpm,
            self.current_rpms[1] / self.max_rpm,
            self.current_rpms[2] / self.max_rpm,
            self.current_rpms[3] / self.max_rpm
        ])
        return state

    def _calculate_reward_and_done(self, state):
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        ang_vel = state[9:12] 
        
        dist_error = np.linalg.norm(self.target_pos - pos)
        
        vel_error = np.linalg.norm(vel)
        
        tilt_error = abs(angles[0]) + abs(angles[1])
        
        spin_error = np.linalg.norm(ang_vel)


        reward = 5.0 
        
        reward -= (1.0 * dist_error)
        reward -= (0.2 * vel_error)
        reward -= (0.5 * tilt_error)
        reward -= (0.1 * spin_error)
        
  
        if dist_error < 0.1 and vel_error < 0.2:
            reward += 2.0

        done = False
        if pos[2] < 0.2: 
            reward -= 100
            done = True
        elif dist_error > 2.0:
            reward -= 50
            done = True
            
        return reward, done