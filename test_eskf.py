import numpy as np
import matplotlib.pyplot as plt

from config import default_config
from drone_dynamics import QuadrotorDynamics
from sensors import SensorSimulator
from eskf import ESKF
from trajectory import TrajectoryGenerator
from controller import CascadedController


def test_eskf_closed_loop():
    print("--- Test: ESKF tijdens gecontroleerde 3D vlucht ---")

    # 1. Initialiseer alle componenten van Fase 1 t/m 5!
    drone = QuadrotorDynamics(default_config)
    controller = CascadedController(default_config)
    sensors = SensorSimulator(dt_imu=0.01, dt_gps=0.1)
    eskf = ESKF()

    # Vlieg een mooie curve van oorsprong naar (X=5, Y=5, Z=5) in 6 seconden
    planner = TrajectoryGenerator(np.array([0, 0, 0]), np.array([5.0, 5.0, 5.0]), 6.0)

    dt = 0.01
    t_array = np.arange(0, 6.0, dt)

    # Data opslag
    history_true = []
    history_gps = []
    history_est = []

    for t in t_array:
        # --- A. BESTURING ---
        # De controller gebruikt (voor nu even) de ware positie om te sturen,
        # zodat we zeker weten dat de vlucht soepel is.
        ref = planner.get_reference(t)
        motor_cmds = controller.compute_control(drone.state, ref)

        # --- B. FYSICA SIMULATIE ---
        drone.step(dt, motor_cmds)

        # --- C. SENSOREN (De 'Echte' wereld meten) ---
        # Bereken wat de IMU zou moeten voelen (Specifieke kracht in Body frame)
        true_acc_world = drone.calculate_derivatives(drone.state, motor_cmds)[3:6] + np.array([0, 0, default_config.g])
        R_true = drone.get_rotation_matrix(drone.state[6:10])
        true_acc_body = R_true.T @ true_acc_world
        true_gyro_body = drone.state[10:13]

        # Voeg ruis toe
        acc_meas, gyro_meas = sensors.get_imu_measurement(true_acc_body, true_gyro_body)
        has_gps, gps_meas = sensors.get_gps_measurement(drone.state[0:3], dt)

        # --- D. NAVIGATIE (ESKF) ---
        # Voorspel met IMU
        eskf.predict_imu(acc_meas, gyro_meas, dt)

        # Corrigeer met GPS
        if has_gps:
            eskf.update_gps(gps_meas)
            history_gps.append((gps_meas[0], gps_meas[1]))  # Sla X, Y op

        # Sla op voor de plot
        history_true.append(drone.state[0:3].copy())
        history_est.append(eskf.nom_state[0:3].copy())

    # Omzetten naar Numpy arrays voor plotten
    history_true = np.array(history_true)
    history_est = np.array(history_est)
    history_gps = np.array(history_gps)

    # --- 2D VISUALISATIE (Bovenaanzicht) ---
    plt.figure(figsize=(10, 8))

    plt.plot(history_true[:, 0], history_true[:, 1], label='Ground Truth (Echte Pad)', color='blue', linewidth=3)
    plt.scatter(history_gps[:, 0], history_gps[:, 1], label='Noisy GPS Metingen', color='red', s=30, alpha=0.5,
                marker='x')
    plt.plot(history_est[:, 0], history_est[:, 1], label='ESKF Schatting', color='green', linewidth=2, linestyle='--')

    plt.title("ESKF Sensor Fusie in Gecontroleerde Vlucht")
    plt.xlabel("X Positie (m)")
    plt.ylabel("Y Positie (m)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_eskf_closed_loop()