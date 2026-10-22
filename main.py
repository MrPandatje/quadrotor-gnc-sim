import matplotlib
matplotlib.use('macosx')  # Forceert lokaal een interactief Mac-venster


import numpy as np
from config import default_config
from drone_dynamics import QuadrotorDynamics
from trajectory import TrajectoryGenerator
from controller import CascadedController
from visualize import DroneVisualizer


def run_simulation():
    print("🚀 Welkom bij de 3D Quadrotor GNC Simulatie!")

    # 1. Setup
    drone = QuadrotorDynamics(default_config)
    controller = CascadedController(default_config)

    # We vliegen een mooi parcours: Omhoog, dan naar voren, dan naar rechts
    start = np.array([0.0, 0.0, 0.0])
    doel = np.array([3.0, 3.0, 5.0])
    vlieg_tijd = 8.0
    planner = TrajectoryGenerator(start, doel, vlieg_tijd, start_yaw=0.0, end_yaw=np.pi / 2)

    dt = 0.02
    t_array = np.arange(0, vlieg_tijd + 2.0, dt)  # 2 sec extra hoveren op het einde

    geschiedenis_pos = []
    geschiedenis_q = []
    geschiedenis_ref = []

    print("Berekent vlucht... Even geduld.")

    # 2. Control Loop
    for t in t_array:
        ref = planner.get_reference(t)

        # Controller berekent input
        motor_cmds = controller.compute_control(drone.state, ref)

        # Fysica update
        drone.step(dt, motor_cmds)

        # Opslaan voor animatie
        geschiedenis_pos.append(drone.state[0:3].copy())
        geschiedenis_q.append(drone.state[6:10].copy())
        geschiedenis_ref.append(ref['pos'].copy())

    geschiedenis_pos = np.array(geschiedenis_pos)
    geschiedenis_q = np.array(geschiedenis_q)
    geschiedenis_ref = np.array(geschiedenis_ref)

    print("Vlucht berekend! Start visualisatie...")

    # 3. Animatie
    vis = DroneVisualizer(default_config, geschiedenis_pos, geschiedenis_q, geschiedenis_ref)

    # Zet save_gif=True en geef het pad naar jouw Results map op
    vis.animate(dt=dt, save_gif=True, filename="Results/3d_vlucht.gif")


if __name__ == "__main__":
    run_simulation()