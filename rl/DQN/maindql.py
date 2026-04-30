import numpy as np
import matplotlib.pyplot as plt
from dql_quad_env import DQLQuadEnv
from dql_agent import DQLAgent
import torch

EPISODES = 1000
MAX_STEPS = 2000 
CURRICULUM_SWITCH = 3000 

def train():
    env = DQLQuadEnv()
    agent = DQLAgent(state_size=env.state_space_size, action_size=env.action_space_size)
    
    reward_history = []

    print("--- Starting Phase 1: Normal Hover Training ---")
    for e in range(EPISODES):
        fault_active = e >= CURRICULUM_SWITCH
        if e == CURRICULUM_SWITCH:
            print("\n--- Starting Phase 2: Fault Injection Training ---")

        state = env.reset(fault_enabled=fault_active)
        total_reward = 0

        for time_step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.store(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        agent.decay_epsilon()
        if e % agent.target_update_freq == 0:
            agent.update_target_network()

        reward_history.append(total_reward)
        
        if e % 10 == 0:
            print(f"Episode: {e}/{EPISODES} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Fault: {fault_active}")

    torch.save(agent.policy_net.state_dict(), "dql_quad_weights.pth")
    print("Training Complete. Weights saved.")
    
    return agent, env

def evaluate_and_plot(agent, env):
    print("\n--- Running Evaluation Episode for Presentation Data ---")
    
    agent.epsilon = 0.0 
    state = env.reset(fault_enabled=True)
    
    history = {
        'time': [], 'z': [], 'rpms': [], 'roll': [], 'pitch': []
    }
    
    for time_step in range(MAX_STEPS):
        t = time_step * env.dt
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        history['time'].append(t)
        history['z'].append(state[2])
        history['rpms'].append(env.current_rpms.copy())
        history['roll'].append(np.degrees(state[6]))
        history['pitch'].append(np.degrees(state[7]))
        
        state = next_state
        if done:
            print(f"Evaluation finished early at t={t:.2f}s")
            break

    for key in history:
        history[key] = np.array(history[key])

    # --- Generate Presentation Plots ---
    plt.figure(figsize=(12, 10))

    # 1. Altitude Plot (The "Money Plot")
    plt.subplot(3, 1, 1)
    plt.plot(history['time'], history['z'], label='DQL Altitude', color='blue')
    plt.axvline(x=2.0, color='red', linestyle='--', label='50% Power Loss')
    plt.title('Altitude vs. Time')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)

    # 2. Control Effort Plot (To show the "chatter")
    plt.subplot(3, 1, 2)
    for i in range(4):
        plt.plot(history['time'], history['rpms'][:, i], label=f'Rotor {i+1}')
    plt.axvline(x=2.0, color='red', linestyle='--')
    plt.title('Motor RPMs vs. Time')
    plt.ylabel('RPM')
    plt.legend()
    plt.grid(True)

    # 3. Attitude Plot
    plt.subplot(3, 1, 3)
    plt.plot(history['time'], history['roll'], label='Roll (\u03d5)')
    plt.plot(history['time'], history['pitch'], label='Pitch (\u03b8)')
    plt.axvline(x=2.0, color='red', linestyle='--')
    plt.title('Attitude vs. Time')
    plt.ylabel('Degrees')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('presentation_plots.png', dpi=300)
    print("Saved 'presentation_plots.png' to your directory.")

if __name__ == "__main__":
    trained_agent, final_env = train()
    evaluate_and_plot(trained_agent, final_env)