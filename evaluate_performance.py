import time
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.failure_aviary import FailureAviary
from ctrl.str_controller import STRController
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def run_simulation(controller_type, model_path=None, failed_power=0.8):
    # Simulation parameters
    DURATION_SEC = 15
    GUI = False
    NUM_DRONES = 1
    CTRL_FREQ = 240
    FAILURE_TIME = 5.0
    TARGET_POS = np.array([0, 0, 1.0])
    
    # Instantiate environment
    env = FailureAviary(gui=GUI, num_drones=NUM_DRONES, ctrl_freq=CTRL_FREQ)
    drone_mass = env.M
    
    # Instantiate controller
    if controller_type == "Baseline":
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    elif controller_type == "STR" or controller_type == "Hybrid":
        ctrl = STRController(drone_model=DroneModel.CF2X, 
                             lambda_factor=0.99,
                             kf=env.KF,
                             km=env.KM,
                             arm_length=env.L,
                             mass=env.M,
                             inertia=env.J)
        if controller_type == "Hybrid" and model_path and os.path.exists(model_path):
            rl_agent = PPO.load(model_path)
            print(f"Loaded RL model from {model_path}")
        else:
            if controller_type == "Hybrid":
                print("Warning: RL model not found. Using fixed parameters for Hybrid case.")
            rl_agent = None
            
    # Reset environment
    obs, info = env.reset()
    
    # Logging
    history = {
        'time': [],
        'z': [],
        'rpy': [],
        'residual': [],
        'k_hat': []
    }
    
    print(f"Running simulation for {controller_type} (Power left: {failed_power*100:.1f}%)...")
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        t = i / CTRL_FREQ
        cur_obs = obs[0]
        
        # Trigger failure: partial or total failure
        if t >= FAILURE_TIME and np.all(env.failure_mask[0] == 1.0):
            env.fail_motor(0, 1, failed_power=failed_power)
            
            
        # RL action for Hybrid
        if controller_type == "Hybrid" and rl_agent:
            # Construct RL observation (18 dims)
            pos_error = env.target_pos - cur_obs[0:3]
            vel = cur_obs[10:13]
            quat = cur_obs[3:7]
            ang_v = cur_obs[13:16]
            eff = ctrl.get_effectiveness()
            rl_obs = np.hstack([pos_error, vel, quat, ang_v, eff, ctrl.rls.lam]).astype(np.float32)
            
            action, _ = rl_agent.predict(rl_obs, deterministic=True)
            # Map actions
            new_lambda = 0.9 + (action[0] + 1) / 2 * 0.099
            ctrl.rls.lam = new_lambda
            gain_m = 0.5 + (action[1:] + 1) / 2 * 1.5
            ctrl.set_gain_multipliers(gain_m)
            
        # Compute Control
        if controller_type == "Baseline":
            action_rpms, pos_e, yaw_e = ctrl.computeControl(
                control_timestep=1.0/CTRL_FREQ,
                cur_pos=cur_obs[0:3],
                cur_quat=cur_obs[3:7],
                cur_vel=cur_obs[10:13],
                cur_ang_vel=cur_obs[13:16],
                target_pos=TARGET_POS
            )
        else:
            # STR and Hybrid use enhanced adaptive control
            action_rpms, pos_e, yaw_e = ctrl.computeControl(
                control_timestep=1.0/CTRL_FREQ,
                cur_pos=cur_obs[0:3],
                cur_quat=cur_obs[3:7],
                cur_vel=cur_obs[10:13],
                cur_ang_vel=cur_obs[13:16],
                target_pos=TARGET_POS,
                observed_accel_z=cur_obs[24],
                observed_ang_accel=cur_obs[25:28],
                mass=drone_mass
            )
            
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_rpms.reshape(1, 4))
        
        # Log
        history['time'].append(t)
        history['z'].append(cur_obs[2])
        history['rpy'].append(cur_obs[7:10])
        if controller_type != "Baseline":
            history['residual'].append(ctrl.rls.get_residual())
            history['k_hat'].append(ctrl.rls.get_estimates().copy())
        else:
            history['residual'].append(0)
            history['k_hat'].append(np.ones(4))
            
        if terminated or truncated:
            break
            
    env.close()
    history['rpy'] = np.array(history['rpy'])
    history['k_hat'] = np.array(history['k_hat'])
    return history

def evaluate(failed_power=0.8):
    best_model_path = "./models/supervisor/best_model.zip"
    
    results = {}
    results['Baseline'] = run_simulation("Baseline", failed_power=failed_power)
    results['STR'] = run_simulation("STR", failed_power=failed_power)
    # results['Hybrid'] = run_simulation("Hybrid", best_model_path, failed_power=failed_power) # Uncomment when trained
    
    # Plotting
    plt.figure(figsize=(12, 12))
    
    # Altitude comparison
    plt.subplot(3, 1, 1)
    for name, hist in results.items():
        plt.plot(hist['time'], hist['z'], label=name)
    plt.axvline(x=5.0, color='r', linestyle='--', label='Failure')
    plt.ylabel('Altitude (m)')
    plt.title(f'Performance Comparison: Motor 1 Failure (Power left: {failed_power*100:.1f}%)')
    plt.legend()
    plt.grid(True)
    
    # Roll comparison
    plt.subplot(3, 1, 2)
    for name, hist in results.items():
        plt.plot(hist['time'], np.degrees(hist['rpy'][:, 0]), label=f'{name} Roll')
    plt.axvline(x=5.0, color='r', linestyle='--')
    plt.ylabel('Roll Angle (deg)')
    plt.legend()
    plt.grid(True)
    
    # Pitch comparison
    plt.subplot(3, 1, 3)
    for name, hist in results.items():
        plt.plot(hist['time'], np.degrees(hist['rpy'][:, 1]), label=f'{name} Pitch')
    plt.axvline(x=5.0, color='r', linestyle='--')
    plt.ylabel('Pitch Angle (deg)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Evaluation complete. Results saved to performance_comparison.png")

if __name__ == "__main__":
    evaluate()
