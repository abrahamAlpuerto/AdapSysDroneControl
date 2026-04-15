import time
import numpy as np
from env.failure_aviary import FailureAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def test_failure_aviary():
    # Instantiate environment
    env = FailureAviary(gui=False, num_drones=1)
    
    # Check observation space
    print(f"Observation space shape: {env.observation_space.shape}")
    if env.observation_space.shape == (1, 24):
        print("Success: Observation space shape is correct (1, 24)")
    else:
        print(f"Error: Observation space shape is {env.observation_space.shape}, expected (1, 24)")
        
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Check initial failure mask (should be all 1s)
    failure_mask_obs = obs[0, 20:]
    print(f"Initial failure mask in obs: {failure_mask_obs}")
    if np.all(failure_mask_obs == 1.0):
        print("Success: Initial failure mask is correct")
    else:
        print("Error: Initial failure mask is incorrect")
        
    # Apply a hover action and check RPMs
    hover_rpm = np.array([[14500, 14500, 14500, 14500]])
    processed_action = env._preprocessAction(hover_rpm)
    print(f"Processed action (no failure): {processed_action}")
    if np.all(processed_action == 14500):
        print("Success: Processed action is correct without failure")
    else:
        print("Error: Processed action is incorrect without failure")
        
    # Trigger motor failure
    env.fail_motor(drone_index=0, motor_index=1)
    
    # Check failure mask after failure
    obs, reward, terminated, truncated, info = env.step(hover_rpm)
    failure_mask_obs = obs[0, 20:]
    print(f"Failure mask in obs after failure: {failure_mask_obs}")
    if failure_mask_obs[1] == 0.0 and failure_mask_obs[0] == 1.0:
        print("Success: Failure mask is updated correctly in observation")
    else:
        print("Error: Failure mask is NOT updated correctly in observation")
        
    # Check processed action after failure
    processed_action_after = env._preprocessAction(hover_rpm)
    print(f"Processed action (with failure): {processed_action_after}")
    if processed_action_after[0, 1] == 0.0:
        print("Success: Motor 1 RPM is 0 after failure")
    else:
        print("Error: Motor 1 RPM is NOT 0 after failure")
        
    env.close()

if __name__ == "__main__":
    test_failure_aviary()
