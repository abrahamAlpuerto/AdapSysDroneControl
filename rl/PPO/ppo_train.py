from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ppo_quad_env import PPOQuadEnv
import os

class CurriculumCallback(BaseCallback): #turns motor back to fault
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

    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate=3e-4, 
                n_steps=2048, 
                batch_size=64, 
                n_epochs=10, 
                gamma=0.99, 
                device="cuda")
    
    TOTAL_TIMESTEPS = 750_000
    FAULT_SWITCH = 5_000_000
    
    callback = CurriculumCallback(switch_step=FAULT_SWITCH)
    
    print("Starting PPO Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    
    # Save the trained model
    os.makedirs("./models", exist_ok=True)
    model.save("./models/ppo_quad_adaptive_v6")
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_ppo()