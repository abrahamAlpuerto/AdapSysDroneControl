import pybullet as p
import time
import torch
from dql_quad_env import DQLQuadEnv
from dql_agent import DQLAgent

def watch_trained_drone():
    print("--- Loading Environment with GUI ---")
    # Initialize the environment with the GUI turned on
    env = DQLQuadEnv(gui=True)
    
    # Initialize the agent architecture
    agent = DQLAgent(state_size=env.state_space_size, action_size=env.action_space_size)
    
    # Load the trained PyTorch weights
    try:
        # Map location ensures it loads even if you trained on GPU but watch on CPU
        agent.policy_net.load_state_dict(torch.load("dql_quad_weights.pth", map_location=agent.device))
        agent.policy_net.eval() # Set network to evaluation mode
        print("Successfully loaded 'dql_quad_weights.pth'")
    except FileNotFoundError:
        print("Error: Could not find 'dql_quad_weights.pth'. Ensure the training script finished.")
        return

    # Force greedy actions (0% random exploration)
    agent.epsilon = 0.0 
    
    print("\n--- Starting Simulation ---")
    print("Watch the PyBullet window. Fault injects at t = 4.0s")
    
    # Reset environment and ensure the fault is enabled for this run
    state = env.reset(fault_enabled=True)
    done = False
    
    # Max steps = 1000 (~4 seconds of flight time at 240Hz)
    for time_step in range(2000):
        # IMPORTANT: Throttle the loop so it plays in real-time (1/240th of a second)
        time.sleep(env.dt) 
        
        # Get action from the trained network
        action = agent.act(state)
        
        # Step the physics engine
        next_state, reward, done = env.step(action)
        
        # Print a visual cue in the terminal exactly when the fault hits
        if time_step == int(2.0 / env.dt): 
            print("⚠️ FAULT INJECTED: Rotor 1 dropped to 50% power!")
            
        state = next_state
        
        if done:
            print(f"Simulation ended early (crashed or out of bounds) at t={time_step * env.dt:.2f}s")
            break
            
    print("Simulation finished. Closing in 3 seconds...")
    time.sleep(3)
    p.disconnect()

if __name__ == "__main__":
    watch_trained_drone()