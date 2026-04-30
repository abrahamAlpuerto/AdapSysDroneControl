# Hybrid Adaptive Control for Quadrotor Motor Failure

This project implements an **Adaptive STR controller** and **Reinforcement Learning controller** for quadrotors to handle partial motor failures in real-time.

## Project Structure
- `env/failure_aviary.py`: Custom PyBullet environment with motor failure and extended observations.
- `ctrl/rls_estimator.py`: Recursive Least Squares for real-time motor effectiveness identification.
- `ctrl/str_controller.py`: Adaptive STR on top of PID with control redistribution.
- `rl/DQN/dql_quad_env.py`: RL environment for training the DQN agent.
- `rl/DQN/dql_agent.py`: DQN agent.
- `rl/DQN/maindql.py`: Script to test the DQN agent under motor faiture.
- `rl/PPO/ppo_quad_env.py`: RL environment for training the PPO agent.
- `rl/PPO/ppo_train.py`: Train initial PPO agent.
- `rl/PPO/transfer_learning.py`: Transfer the learning from initial PPO agent to situation with greater failure margin.
- `rl/PPO/get_graphs.py`: Script to test the PPO agent and compare with baselines under motor faiture.
- `test_str.py`: Script to test the standalone STR controller under motor failure.
- `evaluate_performance.py`: Compare between baseline PID and STR controllers.
