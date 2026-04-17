import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ppo_quad_env import PPOQuadEnv

def evaluate_and_plot():
    models_to_test = {
        "Baseline (No Fault Training)": "./models/ppo_quad_adaptive_v6",
        "Baseline 2nd(No Fault Training)": "./models/ppo_quad_adaptive_v6.1",
        "Baseline 3rd(No Fault Training)": "./models/best_model",
        "Transfer Learning Layer": "./models/transfer_learning_best/best_model",
        "Transfer Learning Layer 2": "./models/transfer_learning_best_v2/best_model",
        "Transfer Learning Layer 3 w/ changed reward function": "./models/transfer_learning_best_v3/best_model",
    }


    env = PPOQuadEnv(gui=False)
    
    model_data = {}
    
    total_steps = 2000  
    dt = env.dt
    time_axis = np.linspace(0, total_steps * dt, total_steps)


    for model_name, model_path in models_to_test.items():
        print(f"Testing: {model_name}...")
        
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"  [!] Could not load {model_path}. Skipping. Error: {e}")
            continue

        obs, _ = env.reset(options={'fault_enabled': True})
        
        altitudes = []
        rolls = []
        pitches = []
        
        for step in range(total_steps):

            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # The state array is: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r, rpm1...]
            altitudes.append(obs[2]) 
            rolls.append(np.degrees(obs[6]))
            pitches.append(np.degrees(obs[7]))
            



            if terminated or truncated:
                remaining_steps = total_steps - step - 1
                altitudes.extend([np.nan] * remaining_steps)
                rolls.extend([np.nan] * remaining_steps)
                pitches.extend([np.nan] * remaining_steps)
                print(f"  -> Crashed at {step * dt:.2f} seconds.")
                break
                
        # store the data
        model_data[model_name] = {
            'alt': altitudes,
            'roll': rolls,
            'pitch': pitches
        }


    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#FFC0CB', '#800080', '#00FFFF']

    for i, (model_name, data) in enumerate(model_data.items()):
        color = colors[i % len(colors)]
        
        ax1.plot(time_axis, data['alt'], label=model_name, color=color, linewidth=2)
        
        ax2.plot(time_axis, data['roll'], label=f"{model_name} (Roll)", color=color, linewidth=2)

    ax1.axvline(x=4.0, color='red', linestyle='--', label='20% Motor Fault Injected')
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Altitude Recovery Comparison")
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim([0, 1.5])

    ax2.axvline(x=4.0, color='red', linestyle='--')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Roll Angle (degrees)")
    ax2.set_title("Attitude Stability Comparison")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    plt.savefig("presentation_comparison_graph.png", dpi=300)

    
    plt.show()

if __name__ == "__main__":
    evaluate_and_plot()