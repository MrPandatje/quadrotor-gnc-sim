import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from config import QuadrotorConfig


class DroneVisualizer:
    def __init__(self, config: QuadrotorConfig, history_pos: np.ndarray, history_q: np.ndarray,
                 history_ref: np.ndarray = None):
        self.config = config
        self.pos = history_pos
        self.q = history_q
        self.ref = history_ref
        self.num_frames = len(self.pos)

        # Bereken snelheid voor de HUD (door positie af te leiden)
        self.vel = np.gradient(self.pos, axis=0) / 0.02  # dt is 0.02

        self.l = self.config.L * (np.sqrt(2) / 2)

    def _get_R(self, q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

    def _get_drone_arms(self, p: np.ndarray, q: np.ndarray):
        R = self._get_R(q)
        m1 = np.array([self.l, self.l, 0])
        m2 = np.array([-self.l, -self.l, 0])
        m3 = np.array([-self.l, self.l, 0])
        m4 = np.array([self.l, -self.l, 0])

        w1 = p + R @ m1
        w2 = p + R @ m2
        w3 = p + R @ m3
        w4 = p + R @ m4

        arm1 = np.column_stack((w2, w1))
        arm2 = np.column_stack((w3, w4))

        return arm1, arm2, p

    def animate(self, dt: float, save_gif: bool = False, filename: str = "drone_flight.gif"):
        fig = plt.figure(figsize=(12, 9))  # Iets groter venster gemaakt voor de HUD

        # Zwarte achtergrond maakt het veel professioneler!
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e1e')

        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e1e')

        # Assen instellen
        max_range = np.array([self.pos[:, 0].max() - self.pos[:, 0].min(),
                              self.pos[:, 1].max() - self.pos[:, 1].min(),
                              self.pos[:, 2].max() - self.pos[:, 2].min()]).max() / 2.0
        mid_x = (self.pos[:, 0].max() + self.pos[:, 0].min()) * 0.5
        mid_y = (self.pos[:, 1].max() + self.pos[:, 1].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(0, self.pos[:, 2].max() + 1.0)  # Vloer altijd op 0

        ax.set_xlabel('X (Oost) [m]')
        ax.set_ylabel('Y (Noord) [m]')
        ax.set_zlabel('Z (Omhoog) [m]')
        ax.set_title('Quadrotor NMPC & ESKF Simulatie', color='white', pad=20, fontsize=14)

        # Grid lichter maken
        ax.grid(color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Trajecten
        line_traj, = ax.plot([], [], [], color='#00ffcc', linewidth=2, label='Drone Traject')
        if self.ref is not None:
            ax.plot(self.ref[:, 0], self.ref[:, 1], self.ref[:, 2], color='gray', linestyle='--', alpha=0.7,
                    label='Referentie')

        # Nieuw: Schaduw op de vloer (Z=0)
        line_shadow, = ax.plot([], [], [], color='white', linestyle='--', alpha=0.3, linewidth=1.5)

        # Nieuw: Drone body met 'markers' voor propellers
        # Marker 'o' maakt ronde schijfjes op de uiteinden van de lijnen
        arm1_line, = ax.plot([], [], [], color='#ff3333', linewidth=3, marker='o', markersize=10,
                             markeredgecolor='none', alpha=0.8)  # Rode voorkant
        arm2_line, = ax.plot([], [], [], color='#cccccc', linewidth=3, marker='o', markersize=10,
                             markeredgecolor='none', alpha=0.8)  # Grijze achterkant
        body_dot, = ax.plot([], [], [], color='white', marker='o', markersize=12)  # Center Hub

        ax.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='none', labelcolor='white')

        # Nieuw: Telemetry HUD tekst
        time_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, color='#00ffcc', fontsize=12, family='monospace')

        def update(frame):
            # Update Traject
            line_traj.set_data(self.pos[:frame, 0], self.pos[:frame, 1])
            line_traj.set_3d_properties(self.pos[:frame, 2])

            # Update Schaduw
            line_shadow.set_data(self.pos[:frame, 0], self.pos[:frame, 1])
            line_shadow.set_3d_properties(np.zeros(frame))  # Z=0

            # Update Drone Armen en Body
            arm1, arm2, p = self._get_drone_arms(self.pos[frame], self.q[frame])

            arm1_line.set_data(arm1[0, :], arm1[1, :])
            arm1_line.set_3d_properties(arm1[2, :])

            arm2_line.set_data(arm2[0, :], arm2[1, :])
            arm2_line.set_3d_properties(arm2[2, :])

            body_dot.set_data([p[0]], [p[1]])
            body_dot.set_3d_properties([p[2]])

            # Update HUD
            snelheid = np.linalg.norm(self.vel[frame])
            time_text.set_text(f"Tijd    : {frame * dt:.1f} s\nHoogte  : {p[2]:.2f} m\nSnelheid: {snelheid:.1f} m/s")

            return line_traj, line_shadow, arm1_line, arm2_line, body_dot, time_text

        step = max(1, self.num_frames // 200)
        frames = range(0, self.num_frames, step)

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt * 1000 * step, blit=False)

        if save_gif:
            print(f"Bezig met renderen en opslaan van {filename}...")
            writer = animation.PillowWriter(fps=int(1 / (dt * step)))
            ani.save(filename, writer=writer)
            print("✅ Exporteren voltooid! Bekijk je Dark-Mode animatie!")
        else:
            plt.show()