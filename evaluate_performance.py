import time
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.failure_aviary import FailureAviary
from ctrl.str_controller import STRController
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def run_simulation(controller_type, model_path=None):
    # Simulation parameters
    DURATION_SEC = 10
    GUI = False
    NUM_DRONES = 1
    CTRL_FREQ = 240
    FAILURE_TIME = 3.0
    TARGET_POS = np.array([0, 0, 1.0])
    
    # Instantiate environment
    env = FailureAviary(gui=GUI, num_drones=NUM_DRONES, ctrl_freq=CTRL_FREQ)
    drone_mass = env.M
    
    # Instantiate controller
    if controller_type == "Baseline":
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    elif controller_type == "STR":
        ctrl = STRController(drone_model=DroneModel.CF2X, lambda_factor=0.99)
    elif controller_type == "Hybrid":
        ctrl = STRController(drone_model=DroneModel.CF2X, lambda_factor=0.99)
        if model_path and os.path.exists(model_path):
            rl_agent = PPO.load(model_path)
            print(f"Loaded RL model from {model_path}")
        else:
            print("Warning: RL model not found. Using fixed lambda for Hybrid case.")
            rl_agent = None
            
    # Reset environment
    obs, info = env.reset()
    
    # Logging
    history = {
        'time': [],
        'z': [],
        'z_error': [],
        'residual': [],
        'lambda': []
    }
    
    print(f"Running simulation for {controller_type}...")
    
    for i in range(int(DURATION_SEC * CTRL_FREQ)):
        t = i / CTRL_FREQ
        cur_obs = obs[0]
        
        # Trigger failure
        if t >= FAILURE_TIME and np.all(env.failure_mask[0] == 1.0):
            env.fail_motor(0, 1)
            
        # RL action for Hybrid
        if controller_type == "Hybrid" and rl_agent:
            # Construct RL observation (12 dims)
            pos_error = env.target_pos - cur_obs[0:3]
            vel = cur_obs[10:13]
            residual = ctrl.rls.get_residual()
            eff = ctrl.get_effectiveness()
            rl_obs = np.hstack([pos_error, vel, residual, eff, ctrl.rls.lam]).astype(np.float32)
            
            action, _ = rl_agent.predict(rl_obs, deterministic=True)
            # Map action [-1, 1] to [0.9, 0.999]
            new_lambda = 0.9 + (action[0] + 1) / 2 * 0.099
            ctrl.rls.lam = new_lambda
            
        # Compute Control
        if controller_type == "Baseline":
            # Baseline is standard PID, no RLS update
            action_rpms, pos_e, yaw_e = ctrl.computeControl(
                control_timestep=1.0/CTRL_FREQ,
                cur_pos=cur_obs[0:3],
                cur_quat=cur_obs[3:7],
                cur_vel=cur_obs[10:13],
                cur_ang_vel=cur_obs[13:16],
                target_pos=TARGET_POS
            )
        else:
            # STR and Hybrid use adaptive control
            action_rpms, pos_e, yaw_e = ctrl.computeControl(
                control_timestep=1.0/CTRL_FREQ,
                cur_pos=cur_obs[0:3],
                cur_quat=cur_obs[3:7],
                cur_vel=cur_obs[10:13],
                cur_ang_vel=cur_obs[13:16],
                target_pos=TARGET_POS,
                observed_accel_z=cur_obs[24],
                mass=drone_mass
            )
            
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_rpms.reshape(1, 4))
        
        # Log
        history['time'].append(t)
        history['z'].append(cur_obs[2])
        history['z_error'].append(TARGET_POS[2] - cur_obs[2])
        if controller_type != "Baseline":
            history['residual'].append(ctrl.rls.get_residual())
            history['lambda'].append(ctrl.rls.lam)
        else:
            history['residual'].append(0)
            history['lambda'].append(1.0)
            
        if terminated or truncated:
            break
            
    env.close()
    return history

def evaluate():
    best_model_path = "./models/supervisor/best_model.zip"
    
    results = {}
    results['Baseline'] = run_simulation("Baseline")
    results['STR'] = run_simulation("STR")
    results['Hybrid'] = run_simulation("Hybrid", best_model_path)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Altitude comparison
    plt.subplot(2, 1, 1)
    for name, hist in results.items():
        plt.plot(hist['time'], hist['z'], label=name)
    plt.axvline(x=3.0, color='r', linestyle='--', label='Failure')
    plt.ylabel('Altitude (m)')
    plt.title('Controller Performance Comparison under Motor Failure')
    plt.legend()
    plt.grid(True)
    
    # RLS Residual comparison
    plt.subplot(2, 1, 2)
    for name, hist in results.items():
        if name != 'Baseline':
            plt.plot(hist['time'], hist['residual'], label=f'{name} Residual')
    plt.axvline(x=3.0, color='r', linestyle='--')
    plt.ylabel('RLS Residual')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Evaluation complete. Results saved to performance_comparison.png")

if __name__ == "__main__":
    evaluate()
