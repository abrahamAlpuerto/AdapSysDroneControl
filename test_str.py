import time
import numpy as np
import matplotlib.pyplot as plt
from env.failure_aviary import FailureAviary
from ctrl.str_controller import STRController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def test_str_adaptation():
    # Simulation parameters
    DURATION_SEC = 15
    GUI = True
    NUM_DRONES = 1
    CTRL_FREQ = 240
    FAILURE_TIME = 5.0 # seconds
    FAILED_POWER = 0.8 # 0.0 for complete failure, 0.8 for 20% loss
    
    # Instantiate environment
    env = FailureAviary(gui=GUI, 
                        num_drones=NUM_DRONES, 
                        ctrl_freq=CTRL_FREQ,
                        physics=Physics.PYB)
    
    # Get drone properties
    drone_mass = env.M
    
    # Instantiate controller
    ctrl = STRController(drone_model=DroneModel.CF2X, 
                         lambda_factor=0.99,
                         kf=env.KF,
                         km=env.KM,
                         arm_length=env.L,
                         mass=env.M,
                         inertia=env.J)
    
    # Target trajectory: Hover at 1.0m
    target_pos = np.array([0, 0, 1.0])
    
    # Reset environment
    obs, info = env.reset()
    
    # Logging
    history = {
        'time': [],
        'z': [],
        'rpy': [],
        'z_accel': [],
        'ang_accel': [],
        'k_estimates': [],
        'rpms': []
    }
    
    print(f"Starting simulation for {DURATION_SEC}s...")
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        t = i / CTRL_FREQ
        
        # Current state (28 dimensions)
        cur_obs = obs[0]
        cur_pos = cur_obs[0:3]
        cur_quat = cur_obs[3:7]
        cur_rpy = cur_obs[7:10]
        cur_vel = cur_obs[10:13]
        cur_ang_vel = cur_obs[13:16]
        z_accel = cur_obs[24]
        ang_accel = cur_obs[25:28]
        
        # Trigger failure: partial or complete failure of motor 1
        if t >= FAILURE_TIME and np.all(env.failure_mask[0] == 1.0):
            env.fail_motor(drone_index=0, motor_index=1, failed_power=FAILED_POWER)
            print(f"--- Motor Failure Injected at t={t:.2f}s (Power left: {FAILED_POWER*100:.1f}%) ---")
        
        # Compute control
        # manual_k override to test redistribution logic in isolation                                                                                                          │
        k_ground_truth = env.failure_mask[0].copy()

        action_rpms, pos_e, yaw_e = ctrl.computeControl(control_timestep=1.0/CTRL_FREQ,
                                                        cur_pos=cur_pos,
                                                        cur_quat=cur_quat,
                                                        cur_vel=cur_vel,
                                                        cur_ang_vel=cur_ang_vel,
                                                        target_pos=target_pos,
                                                        observed_accel_z=z_accel,
                                                        observed_ang_accel=ang_accel,
                                                        mass=drone_mass,
                                                        manual_k=k_ground_truth)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_rpms.reshape(1, 4))
        
        # Log data
        history['time'].append(t)
        history['z'].append(cur_pos[2])
        history['rpy'].append(cur_rpy.copy())
        history['z_accel'].append(z_accel)
        history['ang_accel'].append(ang_accel.copy())
        history['k_estimates'].append(ctrl.get_rls_estimates().copy())
        history['rpms'].append(action_rpms.copy())
        
        # Record B-matrix
        k_hat = ctrl.get_rls_estimates()
        B_hat = ctrl._compute_B(k_hat)
        if 'b_matrix' not in history: history['b_matrix'] = []
        history['b_matrix'].append(B_hat.copy())
        
        if terminated or truncated:
            print(f"Simulation ended at t={t:.2f}s")
            break
            
    env.close()
    
    # Save history for animation
    np.savez('sim_history.npz', **history)
    print("Simulation history saved to sim_history.npz")
    
    # Plot results
    history['time'] = np.array(history['time'])
    history['z'] = np.array(history['z'])
    history['rpy'] = np.array(history['rpy'])
    history['k_estimates'] = np.array(history['k_estimates'])
    history['rpms'] = np.array(history['rpms'])
    
    plt.figure(figsize=(12, 12))
    
    # Plot Altitude
    plt.subplot(4, 1, 1)
    plt.plot(history['time'], history['z'], label='Altitude (z)')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--', label='Failure')
    plt.ylabel('Height (m)')
    plt.title('Enhanced STR Performance: Total Motor Failure Recovery')
    plt.legend()
    plt.grid(True)
    
    # Plot Attitude (Roll/Pitch)
    plt.subplot(4, 1, 2)
    plt.plot(history['time'], np.degrees(history['rpy'][:, 0]), label='Roll')
    plt.plot(history['time'], np.degrees(history['rpy'][:, 1]), label='Pitch')
    plt.plot(history['time'], np.degrees(history['rpy'][:, 2]), label='Yaw')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True)
    
    # Plot RLS Estimates
    plt.subplot(4, 1, 3)
    for i in range(4):
        plt.plot(history['time'], history['k_estimates'][:, i], label=f'Motor {i}')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--')
    plt.ylabel('Effectiveness (k_hat)')
    plt.legend()
    plt.grid(True)
    
    # Plot RPMs
    plt.subplot(4, 1, 4)
    for i in range(4):
        plt.plot(history['time'], history['rpms'][:, i], label=f'Motor {i}')
    plt.axvline(x=FAILURE_TIME, color='r', linestyle='--')
    plt.ylabel('RPM')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('str_test_results.png')
    print("Results saved to str_test_results.png")

if __name__ == "__main__":
    test_str_adaptation()
