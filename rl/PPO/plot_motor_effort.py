import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ppo_quad_env import PPOQuadEnv




def plot_motor_effort():
    model_path = "./models/transfer_learning_best_v3/best_model"
    

    env = PPOQuadEnv(gui=False)
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    total_steps = 2000
    dt = env.dt
    time_axis = np.linspace(0, total_steps * dt, total_steps)
    

    motor_1, motor_2, motor_3, motor_4 = [], [], [], []

    obs, _ = env.reset(options={'fault_enabled': True})
    
    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        

        motor_1.append(action[0])
        motor_2.append(action[1])
        motor_3.append(action[2])
        motor_4.append(action[3])
        
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            remaining = total_steps - step - 1
            motor_1.extend([np.nan] * remaining)
            motor_2.extend([np.nan] * remaining)
            motor_3.extend([np.nan] * remaining)
            motor_4.extend([np.nan] * remaining)
            break

    plt.figure(figsize=(10, 5))
    
    plt.plot(time_axis, motor_1, label='Motor 1 (Faulted)', color='#d62728', linewidth=2)
    plt.plot(time_axis, motor_2, label='Motor 2', color='#1f77b4', alpha=0.8)
    plt.plot(time_axis, motor_3, label='Motor 3', color='#2ca02c', alpha=0.8)
    plt.plot(time_axis, motor_4, label='Motor 4', color='#ff7f0e', alpha=0.8)

    plt.axvline(x=4.0, color='black', linestyle='--', label='20% Power Loss Injected')
    
    plt.title("Control Output (Best Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("Action Command / Motor Effort")
    plt.grid(True)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig("motor_effort_graph.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_motor_effort()