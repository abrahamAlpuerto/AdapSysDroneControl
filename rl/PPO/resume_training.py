from stable_baselines3 import PPO
from ppo_quad_env import PPOQuadEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

def resume_training():
    env = PPOQuadEnv(gui=False)
    
    env.fault_enabled = False
    

    custom_lr = {"learning_rate": 8e-5}

    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/',
        log_path='./logs/', 
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )
    model = PPO.load("./models/ppo_quad_adaptive_v6.1", env=env, custom_objects=custom_lr)

    model.learn(total_timesteps=250_000, callback=eval_callback)
    

    model.save("./models/ppo_quad_adaptive_v6.2")






def micro_finetune():
    env = PPOQuadEnv(gui=False)
    env.fault_enabled = True 
    

    custom_lr = {"learning_rate": 1e-6}
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/',
        log_path='./logs/', 
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )
    model = PPO.load("./models/ppo_quad_adaptive_v5", env=env, custom_objects=custom_lr)
    model.learn(total_timesteps=500_000) 
    model.save("./models/ppo_fault_tolerant_final")



if __name__ == "__main__":
    resume_training()
    # micro_finetune()