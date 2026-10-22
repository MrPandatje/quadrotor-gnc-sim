import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import default_config
from drone_dynamics import QuadrotorDynamics
from trajectory import TrajectoryGenerator
from controller import CascadedController


def run_closed_loop_simulation():
    print("--- Gesloten-lus Simulatie & Visualisatie ---")

    # 1. Initialiseer componenten
    drone = QuadrotorDynamics(default_config)
    controller = CascadedController(default_config)

    # We plannen een traject van oorsprong naar (X=5, Y=5, Z=10) in 8 seconden
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([5.0, 5.0, 10.0])
    T_flight = 8.0
    planner = TrajectoryGenerator(start_pos, end_pos, T_flight)

    # 2. Simulatie instellingen
    dt = 0.01
    time_steps = int((T_flight + 2.0) / dt)  # Vlieg 2 sec extra om te laten zien dat hij netjes blijft hangen
    t_array = np.linspace(0, T_flight + 2.0, time_steps)

    # Opslag voor de plot
    history_pos = []
    history_ref = []

    print("Simulatie draait... Even geduld aub.")

    # 3. De Control Loop
    for t in t_array:
        # A. Vraag aan de planner: Waar moet ik NU zijn?
        ref = planner.get_reference(t)

        # B. Vraag aan de regelaar: Welke motorsnelheden zijn daarvoor nodig?
        motor_cmds = controller.compute_control(drone.state, ref)

        # C. Stuur de motorsnelheden naar de natuurkunde (Dynamics) en stap de tijd vooruit
        drone.step(dt, motor_cmds)

        # Data opslaan voor visualisatie
        history_pos.append(drone.state[0:3].copy())
        history_ref.append(ref['pos'].copy())

    history_pos = np.array(history_pos)
    history_ref = np.array(history_ref)

    # 4. Controle of hij het doel gehaald heeft
    eind_fout = np.linalg.norm(history_pos[-1] - end_pos)
    if eind_fout < 0.1:
        print(f"✅ SUCCES: Drone is gearriveerd! Eindfout = {eind_fout:.3f} meter.")
    else:
        print(f"❌ FOUT: Drone crasht of mist doel. Eindfout = {eind_fout:.3f} meter.")

    # --- 5. 3D VISUALISATIE ---
    print("Bezig met plotten (sluit het plot-venster om script te beëindigen)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot het "Spook pad" (De theorie)
    ax.plot(history_ref[:, 0], history_ref[:, 1], history_ref[:, 2],
            label='Geplande Traject', color='gray', linestyle='--')

    # Plot de werkelijke vlucht
    ax.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2],
            label='Werkelijke Drone Vlucht', color='blue', linewidth=2)

    # Start en Eindpunt markeren
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=100, label='Start')
    ax.scatter(end_pos[0], end_pos[1], end_pos[2], color='red', s=100, label='Doel')

    # Plot opmaak (ENU frame dus Z = Up)
    ax.set_title("3D Quadrotor Vlucht: Oorsprong naar (5, 5, 10)")
    ax.set_xlabel("X (Oost) [m]")
    ax.set_ylabel("Y (Noord) [m]")
    ax.set_zlabel("Z (Omhoog) [m]")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    run_closed_loop_simulation()