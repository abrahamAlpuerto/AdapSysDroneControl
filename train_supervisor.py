import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from rl.supervisor_env import SupervisorEnv

def train():
    # Create training and evaluation environments
    train_env = SupervisorEnv(gui=False)
    eval_env = SupervisorEnv(gui=False)
    
    # Define the directory to save models and logs
    save_path = "./models/supervisor/"
    os.makedirs(save_path, exist_ok=True)
    
    # Callback to save the best model
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=save_path,
                                 log_path=save_path, 
                                 eval_freq=5000,
                                 deterministic=True, 
                                 render=False)
    
    # Instantiate the agent
    # We use a simple MlpPolicy
    model = PPO("MlpPolicy", 
                train_env, 
                verbose=1, 
                tensorboard_log="./logs/supervisor/",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2)
    
    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=100000, callback=eval_callback)
    
    # Save the final model
    model.save(os.path.join(save_path, "final_model"))
    print(f"Training complete. Models saved to {save_path}")

if __name__ == "__main__":
    train()
