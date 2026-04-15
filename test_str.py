import time
import numpy as np
import matplotlib.pyplot as plt
from env.failure_aviary import FailureAviary
from ctrl.str_controller import STRController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def test_str_adaptation():
    # Simulation parameters
    DURATION_SEC = 10
    GUI = False
    NUM_DRONES = 1
    CTRL_FREQ = 240
    FAILURE_TIME = 3.0 # seconds
    
    # Instantiate environment
    env = FailureAviary(gui=GUI, 
                        num_drones=NUM_DRONES, 
                        ctrl_freq=CTRL_FREQ,
                        physics=Physics.PYB)
    
    # Get drone properties
    drone_mass = env.M
    drone_kf = env.KF
    
    # Instantiate controller
    ctrl = STRController(drone_model=DroneModel.CF2X, 
                         lambda_factor=0.99)
    
    # Target trajectory: Hover at 1.0m
    target_pos = np.array([0, 0, 1.0])
    
    # Reset environment
    obs, info = env.reset()
    
    # Logging
    history = {
        'time': [],
        'z': [],
        'z_accel': [],
        'k_estimates': [],
        'effectiveness': [],
        'rpms': []
    }
    
    print(f"Starting simulation for {DURATION_SEC}s...")
    start_time = time.time()
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        t = i / CTRL_FREQ
        
        # Current state
        # obs[0] shape is (25,)
        # 0:3 pos, 3:7 quat, 7:10 rpy, 10:13 vel, 13:16 ang_v, 16:20 last_action, 20:24 mask, 24: z_accel
        cur_obs = obs[0]
        cur_pos = cur_obs[0:3]
        cur_quat = cur_obs[3:7]
        cur_vel = cur_obs[10:13]
        cur_ang_vel = cur_obs[13:16]
        z_accel = cur_obs[24]
        
        # Trigger failure
        if t >= FAILURE_TIME and np.all(env.failure_mask[0] == 1.0):
            env.fail_motor(drone_index=0, motor_index=1)
            print(f"--- Motor Failure Injected at t={t:.2f}s ---")
        
        # Compute control
        action_rpms, pos_e, yaw_e = ctrl.computeControl(control_timestep=1.0/CTRL_FREQ,
                                                        cur_pos=cur_pos,
                                                        cur_quat=cur_quat,
                                                        cur_vel=cur_vel,
                                                        cur_ang_vel=cur_ang_vel,
                                                        target_pos=target_pos,
                                                        observed_accel_z=z_accel,
                                                        mass=drone_mass)
        
        # Step environment
        # action_rpms is (4,), env expects (NUM_DRONES, 4)
        obs, reward, terminated, truncated, info = env.step(action_rpms.reshape(1, 4))
        
        # Log data
        history['time'].append(t)
        history['z'].append(cur_pos[2])
        history['z_accel'].append(z_accel)
        history['k_estimates'].append(ctrl.get_rls_estimates().copy())
        history['effectiveness'].append(ctrl.get_effectiveness().copy())
        history['rpms'].append(action_rpms.copy())
        
        if terminated or truncated:
            print(f"Simulation ended at t={t:.2f}s")
            break
            
    env.close()
    
    # Plot results
    history['time'] = np.array(history['time'])
    history['z'] = np.array(history['z'])
    history['k_estimates'] = np.array(history['k_estimates'])
    history['effectiveness'] = np.array(history['effectiveness'])
    history['rpms'] = np.array(history['rpms'])
    
    plt.figure(figsize=(12, 10))
    
    # Plot Altitude
    plt.subplot(3, 1, 1)
    plt.plot(history['time'], history['z'], label='Altitude (z)')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--', label='Failure')
    plt.ylabel('Height (m)')
    plt.title('Altitude and RLS Estimates during Motor Failure')
    plt.legend()
    plt.grid(True)
    
    # Plot RLS Estimates
    plt.subplot(3, 1, 2)
    for i in range(4):
        plt.plot(history['time'], history['k_estimates'][:, i], label=f'Motor {i}')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--')
    plt.ylabel('Estimated kf')
    plt.legend()
    plt.grid(True)
    
    # Plot Effectiveness Ratios
    plt.subplot(3, 1, 3)
    for i in range(4):
        plt.plot(history['time'], history['effectiveness'][:, i], label=f'Motor {i}')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--')
    plt.ylabel('Effectiveness Ratio')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('str_test_results.png')
    print("Results saved to str_test_results.png")

if __name__ == "__main__":
    test_str_adaptation()
