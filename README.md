# Hybrid Adaptive Control for Quadrotor Motor Failure

This project implements an **Enhanced Hybrid Control System** for quadrotors to handle complete motor failures in real-time. It combines an **Indirect Self-Tuning Regulator (STR)** with a **Reinforcement Learning (RL) Supervisor**.

## Key Features
- **Enhanced RLS Estimation**: Uses a 4D measurement vector (Vertical and 3D Angular Accelerations) for high-fidelity motor health identification.
- **Pseudo-Inverse Control Mixer**: Dynamically redistributes thrust and moments from failed motors to the remaining healthy ones.
- **RL Supervisor (PPO)**: Optimizes adaptation rates (Forgetting Factor $\lambda$) and tunes PID gain multipliers to handle post-failure flight dynamics.
- **Custom Simulation**: Built on `gym-pybullet-drones` with explicit motor failure injection.

## Project Structure
- `env/failure_aviary.py`: Custom PyBullet environment with motor failure and extended observations.
- `ctrl/rls_estimator.py`: Recursive Least Squares for real-time motor effectiveness identification.
- `ctrl/str_controller.py`: Adaptive PID with control redistribution.
- `rl/supervisor_env.py`: RL environment for training the supervisor.
- `test_str.py`: Script to test the standalone STR controller under total motor failure.
- `train_supervisor.py`: Trains the PPO supervisor agent.
- `evaluate_performance.py`: Comparative analysis between Baseline, STR, and Hybrid controllers.

## Installation & Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8+

### Conda Environment
Activate the pre-configured `drones` environment:
```bash
conda activate drones
```

If the environment is not set up, you can install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Test Standalone STR Adaptation
Run a 15-second simulation where a total failure of Motor 1 is injected at $t=5s$:
```bash
python test_str.py
```
This will generate `str_test_results.png` showing altitude, attitude, and RLS convergence.

### 2. Train the RL Supervisor
Train the PPO agent to optimize the adaptation parameters:
```bash
python train_supervisor.py
```
Logs are saved in `./logs/supervisor/` and models in `./models/supervisor/`.

### 3. Evaluate and Compare
Compare the Baseline (No-Adaptation) vs. STR (Adaptive) vs. Hybrid (STR + RL):
```bash
python evaluate_performance.py
```
Results will be saved in `performance_comparison.png`.

## Success Criteria
The system is considered successful if it recovers from a total motor failure (100% loss) within 2 seconds and maintains altitude/attitude stability for the remainder of the flight.
