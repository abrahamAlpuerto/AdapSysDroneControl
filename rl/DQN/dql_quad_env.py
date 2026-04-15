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
        
        # Action space: 3 options per motor (-100, 0, +100 RPM)
        # 3^4 = 81 discrete actions
        self.action_space_size = 81
        self.state_space_size = 16
        
        # Quadrotor physical parameters
        self.thrust_coeff = 3.16e-10  # k-value for F = k * RPM^2
        self.base_rpm = 15000         # Approximate hover RPM
        self.rpm_step = 100           # How much RPM changes per action
        self.max_rpm = 40000
        self.min_rpm = 0
        
        # Simulation parameters
        self.dt = 1.0 / 240.0
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.reset()

    def reset(self, fault_enabled=False):
        # Store the flag for this episode
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
        """Converts integer 0-80 into 4 RPM adjustments."""
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
        # 1. Update RPMs based on discrete action
        rpm_adjustments = self._decode_action(action_int)
        self.current_rpms += rpm_adjustments
        self.current_rpms = np.clip(self.current_rpms, self.min_rpm, self.max_rpm)
        
        # 2. Calculate Thrust Forces (F = k * RPM^2)
        forces = self.thrust_coeff * (self.current_rpms ** 2)
        
        # 3. FAULT INJECTION: 50% power loss on rotor 1 at t = 4.0s
        t = self.time_step * self.dt
        if t >= 4.0:
            forces[0] *= 0.5 
            
        # 4. Apply physics (applying forces to the 4 rotor links)
        # Note: PyBullet link indices depend on your specific URDF.
        # Assuming links 0, 1, 2, 3 are the rotors.
        for i in range(4):
            p.applyExternalForce(self.quadId, i, 
                                 forceObj=[0, 0, forces[i]], 
                                 posObj=[0, 0, 0], 
                                 flags=p.LINK_FRAME)
            
        p.stepSimulation()
        self.time_step += 1
        
        # 5. Get new state and calculate reward
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
        ang_vel = state[9:12] # NEW: Angular velocity (p, q, r)
        
        # 1. Distance Penalty (L2 Norm is smoother for neural networks than abs values)
        dist_error = np.linalg.norm(self.target_pos - pos)
        
        # 2. Velocity Penalty (Stop it from drifting)
        vel_error = np.linalg.norm(vel)
        
        # 3. Tilt Penalty (Keep it flat - we only care about roll and pitch)
        tilt_error = abs(angles[0]) + abs(angles[1])
        
        # 4. Spin Penalty (Stop it from doing flips or spinning out of control)
        spin_error = np.linalg.norm(ang_vel)

        # --- THE REWARD CALCULATION ---
        # Base survival bonus
        reward = 5.0 
        
        # Subtract weighted penalties
        # You can tune these weights (1.0, 0.2, 0.5, 0.1) if it favors one bad behavior over another
        reward -= (1.0 * dist_error)
        reward -= (0.2 * vel_error)
        reward -= (0.5 * tilt_error)
        reward -= (0.1 * spin_error)
        
        # NEW: The "Bullseye" Bonus (Sparse Reward)
        # If the drone is within 10cm of the target and barely moving, give it a massive bonus.
        # This teaches the network exactly what "perfect" looks like.
        if dist_error < 0.1 and vel_error < 0.2:
            reward += 2.0

        # --- TERMINATION CONDITIONS ---
        done = False
        if pos[2] < 0.2:  # Crashed into the ground
            reward -= 100
            done = True
        elif dist_error > 2.0: # Flew out of bounds
            reward -= 50
            done = True
            
        return reward, done