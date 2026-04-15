from rl.supervisor_env import SupervisorEnv

def test_supervisor_env():
    env = SupervisorEnv()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Take 10 steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Reward={reward:.4f}, Lambda={obs[-1]:.4f}")
        
    env.close()
    print("SupervisorEnv test passed!")

if __name__ == "__main__":
    test_supervisor_env()
