import time
import pybullet as p
from stable_baselines3 import PPO
from ppo_quad_env import PPOQuadEnv


"""created using gemini"""

def watch_ppo_drone():
    print("--- Loading Environment with GUI ---")
    env = PPOQuadEnv(gui=True)
    
    print("--- Loading Trained PPO Model ---")
    try:
        # Point this to the exact name of the file/folder you saved.
        # If it is a zip, you can omit the .zip extension, SB3 finds it automatically.
        model = PPO.load("./models/ppo_quad_adaptive_v6")
        print("Successfully loaded model!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'ppo_quad_adaptive.zip' is in the './models/' directory.")
        return

    print("\n--- Starting Simulation ---")
    print("Watch the PyBullet window. Fault injects at t = 4.0s")
    
    # Reset environment and FORCE the fault to be enabled for this viewing
    obs, info = env.reset(options={'fault_enabled': True})
    
    # Run for 2000 steps (~8.3 seconds) to give plenty of time to see the recovery
    for time_step in range(2000):
        # Throttle the loop so it plays in real-time (1/240th of a second)
        time.sleep(env.dt) 
        
        # Get action from the trained network. 
        # deterministic=True ensures it uses the optimal policy, not random exploration
        action, _states = model.predict(obs, deterministic=False) 
        
        # Step the environment using the Gymnasium API
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print a visual cue in the terminal exactly when the fault hits
        if time_step == int(4.0 / env.dt): 
            print("⚠️ FAULT INJECTED: Rotor 1 dropped to 50% power!")
            
        if terminated or truncated:
            print(f"Simulation ended early at t={time_step * env.dt:.2f}s")
            break
            
    print("Simulation finished. Closing in 3 seconds...")
    time.sleep(3)
    p.disconnect()

if __name__ == "__main__":
    watch_ppo_drone()