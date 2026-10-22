# 3D Quadrotor Guidance, Navigation & Control (GNC) Simulation

[![Status: In Development](https://img.shields.io/badge/status-in_development-orange.svg)](#roadmap)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular computational simulation and control framework for an autonomous quadrotor UAV developed in Python. This repository implements a full Guidance, Navigation, and Control (GNC) stack, incorporating 6-DOF rigid-body dynamics, sensor fusion, and real-time trajectory optimization.

---

> **Project Status:**  
> This project has **just started** and is currently under active development. Core equations of motion, physical parameters, and numerical integrators are being implemented as part of Phase 1.

---

## Architecture Overview

* **6-DOF Dynamics Engine:** Rigid-body Newton-Euler dynamics using unit quaternions (Hamilton convention, scalar-first) to eliminate gimbal lock. Numerical integration via 4th-order Runge-Kutta (RK4).
* **Navigation (State Estimation):** Synthetic sensor generation (IMU with bias/drift, GPS, Barometer) fused via an Error-State Kalman Filter (ESKF).
* **Guidance & Trajectory Generation:** Minimum-snap 3D trajectory generation across waypoints.
* **Control:** 
  * Baseline cascaded geometric attitude and position control.
  * Nonlinear Model Predictive Control (NMPC) formulated in CasADi with IPOPT for trajectory tracking and obstacle avoidance.
* **Disturbances:** Aerodynamic drag and wind gust modeling.

---

## Roadmap

- [x] Repository setup and initial configuration
- [x] **Phase 1:** 6-DOF physical dynamics, quaternion kinematics, and RK4 verification
- [ ] **Phase 2:** 3D minimum-snap trajectory generation
- [ ] **Phase 3:** Cascaded baseline geometric controller
- [ ] **Phase 4:** Nonlinear Model Predictive Control (NMPC) with CasADi
- [ ] **Phase 5:** Sensor modeling and Error-State Kalman Filter (ESKF)
- [ ] **Phase 6:** 3D visualization, benchmarking, and telemetry logging


