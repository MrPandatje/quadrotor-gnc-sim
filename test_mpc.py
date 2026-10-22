import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import default_config
from drone_dynamics import QuadrotorDynamics
from mpc_controller import NonlinearMPC


def test_mpc_flight():
    print("--- Model Predictive Control (NMPC) Simulatie ---")
    print("Let op: CasADi moet iteratief rekenen. Dit kan enkele seconden duren.")

    # Initialisatie
    drone = QuadrotorDynamics(default_config)

    # We kiezen een voorspellingshorizon van 10 stappen van 0.1 seconde (dus hij kijkt 1 seconde vooruit)
    mpc = NonlinearMPC(default_config, N=10, dt=0.1)

    # Doel: Van [0,0,0] rechtstreeks naar [2, 3, 5]
    # State = [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    doel_state = np.array([
        2.0, 3.0, 5.0,  # Positie
        0.0, 0.0, 0.0,  # Snelheid
        1.0, 0.0, 0.0, 0.0,  # Hover oriëntatie (qw=1)
        0.0, 0.0, 0.0  # Geen rotatiesnelheid
    ])

    sim_time = 4.0  # seconden
    dt_sim = 0.1  # We draaien de control loop op 10Hz voor deze demo
    t_array = np.arange(0, sim_time, dt_sim)

    history_pos = []

    for t in t_array:
        # 1. Bepaal motor commando via NMPC
        # De solver berekent het optimale pad in fracties van seconden
        motor_cmds = mpc.compute_control(drone.state, doel_state)

        # 2. Pas toe op de drone dynamica (de 'echte' wereld)
        drone.step(dt_sim, motor_cmds)

        history_pos.append(drone.state[0:3].copy())

        # Print progressie
        print(f"Tijd: {t:.1f}s | Positie: {np.round(drone.state[0:3], 2)}")

    history_pos = np.array(history_pos)

    # --- 3D VISUALISATIE ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], label='NMPC Drone Vlucht', color='purple',
            linewidth=2)
    ax.scatter(0, 0, 0, color='green', s=100, label='Start')
    ax.scatter(doel_state[0], doel_state[1], doel_state[2], color='red', s=100, label='Doel [2, 3, 5]')

    ax.set_title("NMPC: Autonome Point-to-Point Vlucht")
    ax.set_xlabel("X (Oost) [m]")
    ax.set_ylabel("Y (Noord) [m]")
    ax.set_zlabel("Z (Omhoog) [m]")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_mpc_flight()