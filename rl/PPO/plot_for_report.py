import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ppo_quad_env import PPOQuadEnv

def plot_comprehensive():
    # Set the path to your best model
    model_path = "./models/baseline_best_v7/best_model"
    
    print(f"Loading environment and model: {model_path}...")
    env = PPOQuadEnv(gui=False)
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    total_steps = 2000
    dt = env.dt
    time_axis = np.linspace(0, total_steps * dt, total_steps)
    
    # Data storage arrays
    altitudes = []
    rolls = []
    motor_1, motor_2, motor_3, motor_4 = [], [], [], []

    # Force fault to be enabled for this evaluation
    obs, _ = env.reset(options={'fault_enabled': True})
    
    print("Running simulation...")
    for step in range(total_steps):
        # Use deterministic=True for evaluation to see the absolute optimal policy
        action, _ = model.predict(obs, deterministic=True)
        
        # Store actions
        motor_1.append(action[0])
        motor_2.append(action[1])
        motor_3.append(action[2])
        motor_4.append(action[3])
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store state variables (Altitude is index 2, Roll is index 6)
        altitudes.append(obs[2])
        rolls.append(np.degrees(obs[6]))
        
        if terminated or truncated:
            print(f"Simulation ended early at {step * dt:.2f} seconds.")
            # Pad the rest of the arrays with NaNs so the graph drops off cleanly
            remaining = total_steps - step - 1
            altitudes.extend([np.nan] * remaining)
            rolls.extend([np.nan] * remaining)
            motor_1.extend([np.nan] * remaining)
            motor_2.extend([np.nan] * remaining)
            motor_3.extend([np.nan] * remaining)
            motor_4.extend([np.nan] * remaining)
            break

    print("Generating comprehensive plot...")
    
    # Create a 3-row, 1-column figure sharing the X-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    fault_time = 4.0
    fault_label = '50% Thrust Loss Injected'
    
    # --- Subplot 1: Altitude ---
    ax1.plot(time_axis, altitudes, color='#1f77b4', linewidth=2, label='Hovering Baseline')
    ax1.axvline(x=fault_time, color='red', linestyle='--', linewidth=1.5, label=fault_label)
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("System Response to Catastrophic Motor Failure")
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1.set_ylim([0, 2.0])

    # --- Subplot 2: Roll (Attitude) ---
    ax2.plot(time_axis, rolls, color='#2ca02c', linewidth=2)
    ax2.axvline(x=fault_time, color='red', linestyle='--', linewidth=1.5)
    ax2.set_ylabel("Roll Angle (degrees)")
    ax2.grid(True)
    # Give roll a symmetrical limit to show the oscillation clearly
    max_roll = max(np.nanmax(np.abs(rolls)), 10) 
    ax2.set_ylim([-max_roll - 2, max_roll + 2])

    # --- Subplot 3: Motor Effort ---
    ax3.plot(time_axis, motor_1, label='Motor 1 (Faulted)', color='#d62728', linewidth=2)
    ax3.plot(time_axis, motor_2, label='Motor 2', color='#1f77b4', alpha=0.7)
    ax3.plot(time_axis, motor_3, label='Motor 3', color='#2ca02c', alpha=0.7)
    ax3.plot(time_axis, motor_4, label='Motor 4', color='#ff7f0e', alpha=0.7)
    ax3.axvline(x=fault_time, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalized Command [-1, 1]")
    ax3.grid(True)
    ax3.legend(loc='lower left', ncol=4) # Horizontal legend to save vertical space
    ax3.set_ylim([-1.1, 1.1])

    # Adjust layout so everything fits tightly without overlapping
    plt.tight_layout()
    
    # Save in high resolution
    plt.savefig("comprehensive_fault_analysis_first50.png", dpi=300)
    print("Saved graph as 'comprehensive_fault_analysis_first50.png'")
    
    plt.show()

if __name__ == "__main__":
    plot_comprehensive()