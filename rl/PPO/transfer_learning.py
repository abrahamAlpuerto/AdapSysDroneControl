from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from ppo_quad_env import PPOQuadEnv
import torch
import os



def train_added_layer():
    env = PPOQuadEnv(gui=False)
    env.fault_enabled = True 
    eval_env = PPOQuadEnv(gui=False)
    eval_env.fault_enabled = True




    base_model = PPO.load("./models/best_model")
    base_state_dict = base_model.policy.state_dict()


    custom_architecture = dict(net_arch=dict(pi=[64, 64, 32], vf=[64, 64, 32]))
    

    # new_model = PPO("MlpPolicy", 
    #                 env, 
    #                 policy_kwargs=custom_architecture,
    #                 learning_rate=3e-4, 
    #                 device="cuda")   
    
    custom_lr = {"learning_rate": 8e-5}

    new_model = PPO.load("./models/transfer_learning_best_v4/best_model", env=env, custome_objects = custom_lr )


    frozen = 0 
    active = 0


    for name, param in new_model.policy.named_parameters():

        if name in base_state_dict and param.shape == base_state_dict[name].shape:

            param.data.copy_(base_state_dict[name].data)
            param.requires_grad = False
            frozen += 1

            print("Transfered Frozen:", name)

        else:

            param.requires_grad = True
            active += 1 
            print("New Active:", name)

    print(f"Frozen tensor count: {frozen} | Active tensor count: {active}")


    evaluator = EvalCallback(
        eval_env,
        best_model_save_path='./models/transfer_learning_best_v5/',
        log_path='./logs/', 
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )

    new_model.learn(total_timesteps=500_000, callback=evaluator)


if __name__ == "__main__":
    train_added_layer()