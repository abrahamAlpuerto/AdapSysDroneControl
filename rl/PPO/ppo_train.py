from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from ppo_quad_env import PPOQuadEnv
import os

class CurriculumCallback(BaseCallback): 
    def __init__(self, switch_step, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.switch_step = switch_step
        self.fault_active = False

    def _on_step(self) -> bool:
        if self.num_timesteps > self.switch_step and not self.fault_active:
            print(f"\n--- TIMESTEP {self.num_timesteps}: ACTIVATING MOTOR 1 FAULT ---")
            self.fault_active = True
            self.training_env.set_attr('fault_enabled', True)
        return True

def train_ppo():
    env = PPOQuadEnv(gui=False)
    env.fault_enabled = False 

    eval_env = PPOQuadEnv(gui=False)
    eval_env.fault_enabled = False

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO("MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                learning_rate=3e-4, 
                n_steps=2048, 
                batch_size=256,              
                n_epochs=10, 
                gamma=0.99,
                ent_coef=0.01,               
                device="cuda")
    
    TOTAL_TIMESTEPS = 1_500_000
    FAULT_SWITCH = 5_000_000 
    
    curriculum_callback = CurriculumCallback(switch_step=FAULT_SWITCH)
    
    evaluator = EvalCallback(
        eval_env,
        best_model_save_path='./models/baseline_best_v7/',
        log_path='./logs/', 
        eval_freq=10000,     
        deterministic=True,    
        render=False
    )
    
    callbacks = [curriculum_callback, evaluator]
    
    print("Starting Baseline PPO Training with expanded architecture and EvalCallback...")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    
    os.makedirs("./models", exist_ok=True)
    model.save("./models/ppo_quad_baseline_final_v7")
    print("Training complete. Best model saved in './models/baseline_best_v7/'.")

if __name__ == "__main__":
    train_ppo()