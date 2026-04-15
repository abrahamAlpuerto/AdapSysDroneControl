# Project Report Draft (LaTeX Format)

This file contains a LaTeX draft of the project report. You can copy the content below into an `.tex` file in Overleaf.

```latex
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{bm}

\title{Hybrid Adaptive Control for Quadrotor Motor Failure Recovery}
\author{Win Sa}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report outlines the development and evaluation of a hybrid adaptive control system designed to maintain quadrotor stability following a complete motor failure. We implement an Indirect Self-Tuning Regulator (STR) that utilizes Recursive Least Squares (RLS) for real-time parameter estimation, combined with a Reinforcement Learning (RL) supervisor that dynamically tunes the adaptation rate. The system is validated in the \texttt{gym-pybullet-drones} simulation environment, comparing performance against a non-adaptive PID baseline.
\end{abstract}

\section{Project Formulation and Motivation}
Quadrotors are inherently under-actuated and unstable systems. A complete failure of a single motor typically leads to a catastrophic loss of control in standard PID-based systems. Motivation for this project stems from the need for high-reliability autonomous flight in safety-critical applications (e.g., search and rescue, urban delivery). By combining classical adaptive control with modern reinforcement learning, we aim to achieve rapid recovery and robust trajectory tracking even under partial or total loss of control effectiveness.

\section{Simulation Environment}
The project utilizes the \texttt{gym-pybullet-drones} library, a physics-based simulation environment built on the PyBullet engine. A custom environment, \texttt{FailureAviary}, was developed to simulate hardware faults by intercepting motor commands and applying a \textit{failure mask} $M \in \{0, 1\}^4$. The environment provides high-frequency IMU data and world-frame accelerations necessary for real-time estimation.

\section{Drone Dynamics}
The vertical dynamics of the quadrotor in the world frame are modeled as:
\begin{equation}
    m \ddot{z} = \sum_{i=1}^{4} T_i \cos(\phi) \cos(\theta) - mg
\end{equation}
where $T_i = k_f \omega_i^2$ is the thrust produced by motor $i$, $\omega_i$ is the angular velocity (RPM), and $k_f$ is the thrust coefficient. 

For the purpose of the RLS estimator, we assume a near-hover state where the attitude angles $\phi, \theta \approx 0$, leading to the simplified linear-in-parameters model:
\begin{equation}
    a_z + g = \sum_{i=1}^{4} \left( \frac{\omega_i^2}{m} \right) k_i
\end{equation}
where $k_i$ is the effective thrust coefficient of motor $i$. In a healthy state, $k_i = k_{f, \text{nominal}}$. In a failure state, $k_i \to 0$.

\section{Control Methodologies}

\subsection{Baseline: Standard PID}
The baseline controller is a Proportional-Integral-Derivative (PID) controller (\texttt{DSLPIDControl}) tuned for nominal flight conditions. It assumes all motors are fully functional and lacks any mechanism to detect or compensate for loss of thrust.

\subsection{STR: Indirect Self-Tuning Regulator}
The STR implements an Indirect Adaptive Control scheme consisting of two parts:
\begin{enumerate}
    \item \textbf{RLS Estimator:} Estimates $\hat{k}_i$ recursively:
    \begin{equation}
        y(k) = \bm{\phi}^T(k) \hat{\bm{\theta}}(k-1) + \epsilon(k)
    \end{equation}
    where $\bm{\phi} = [\omega_1^2/m, \dots, \omega_4^2/m]^T$.
    \item \textbf{Adaptive Law:} The commanded RPMs are scaled by the inverse square root of the estimated effectiveness:
    \begin{equation}
        \omega_{i, \text{adapted}} = \frac{\omega_{i, \text{nominal}}}{\sqrt{\hat{k}_i / k_{f, \text{nominal}}}}
    \end{equation}
\end{enumerate}

\subsection{Hybrid: RL Supervisor}
The Hybrid controller introduces a PPO-based RL agent that observes the RLS residuals $\epsilon(k)$ and the tracking error $\bm{e}_p$. The agent outputs the optimal forgetting factor $\lambda$ for the RLS estimator:
\begin{itemize}
    \item \textbf{High $\lambda$ (e.g., 0.999):} High noise rejection, slow adaptation.
    \item \textbf{Low $\lambda$ (e.g., 0.90):} Rapid adaptation to sudden changes, but sensitive to measurement noise.
\end{itemize}

\section{Proposed Results Outline}
The final results will include a comparative analysis across three scenarios:
\begin{enumerate}
    \item \textbf{Altitude Retention:} Time-series plots showing altitude drop during failure at $t=3s$.
    \item \textbf{Estimation Convergence:} Comparison of how quickly $\hat{k}_i$ converges to zero using fixed $\lambda$ vs. RL-tuned $\lambda$.
    \item \textbf{Control Effort:} Analysis of the motor RPM saturation during the recovery phase.
\end{enumerate}

\section{Conclusion}
The integration of an RL supervisor allows the STR to transition between a steady-state "precision" mode and a post-failure "emergency adaptation" mode, potentially reducing recovery time and minimizing altitude loss compared to fixed-hyperparameter adaptive schemes.

\end{document}
```
