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

        # Geometrie van de drone voor het tekenen (X-configuratie)
        # l = L * sin(45 deg)
        self.l = self.config.L * (np.sqrt(2) / 2)

    def _get_R(self, q: np.ndarray) -> np.ndarray:
        """Zet quaternion om naar rotatiematrix."""
        qw, qx, qy, qz = q / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

    def _get_drone_arms(self, p: np.ndarray, q: np.ndarray):
        """Berekent de 3D coördinaten van de drone armen."""
        R = self._get_R(q)

        # Lokale motor posities in body frame
        m1 = np.array([self.l, self.l, 0])  # Rechts-Voor
        m2 = np.array([-self.l, -self.l, 0])  # Links-Achter
        m3 = np.array([-self.l, self.l, 0])  # Links-Voor
        m4 = np.array([self.l, -self.l, 0])  # Rechts-Achter

        # Roteer naar wereld frame en tel positie erbij op
        w1 = p + R @ m1
        w2 = p + R @ m2
        w3 = p + R @ m3
        w4 = p + R @ m4

        # Arm 1 (M2 naar M1), Arm 2 (M3 naar M4)
        arm1 = np.column_stack((w2, w1))
        arm2 = np.column_stack((w3, w4))

        return arm1, arm2

    def animate(self, dt: float, save_gif: bool = False, filename: str = "drone_flight.gif"):
        """Maakt de 3D animatie."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Bepaal de grenzen van de plot zodat hij niet verspringt
        max_range = np.array([self.pos[:, 0].max() - self.pos[:, 0].min(),
                              self.pos[:, 1].max() - self.pos[:, 1].min(),
                              self.pos[:, 2].max() - self.pos[:, 2].min()]).max() / 2.0
        mid_x = (self.pos[:, 0].max() + self.pos[:, 0].min()) * 0.5
        mid_y = (self.pos[:, 1].max() + self.pos[:, 1].min()) * 0.5
        mid_z = (self.pos[:, 2].max() + self.pos[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X (Oost) [m]')
        ax.set_ylabel('Y (Noord) [m]')
        ax.set_zlabel('Z (Omhoog) [m]')
        ax.set_title('3D Quadrotor GNC Simulatie')

        # Plot elementen
        line_traj, = ax.plot([], [], [], color='blue', alpha=0.5, label='Gevlogen Traject')
        if self.ref is not None:
            ax.plot(self.ref[:, 0], self.ref[:, 1], self.ref[:, 2], color='gray', linestyle='--',
                    label='Gepland Traject')

        arm1_line, = ax.plot([], [], [], color='red', linewidth=3)
        arm2_line, = ax.plot([], [], [], color='black', linewidth=3)

        ax.legend()

        def update(frame):
            # Update het spoor (de lijn achter de drone)
            line_traj.set_data(self.pos[:frame, 0], self.pos[:frame, 1])
            line_traj.set_3d_properties(self.pos[:frame, 2])

            # Update de drone armen
            arm1, arm2 = self._get_drone_arms(self.pos[frame], self.q[frame])

            arm1_line.set_data(arm1[0, :], arm1[1, :])
            arm1_line.set_3d_properties(arm1[2, :])

            arm2_line.set_data(arm2[0, :], arm2[1, :])
            arm2_line.set_3d_properties(arm2[2, :])

            return line_traj, arm1_line, arm2_line

        # Subsample frames als de simulatie te veel stappen heeft (bijv > 500)
        step = max(1, self.num_frames // 200)
        frames = range(0, self.num_frames, step)

        # interval = milliseconden per frame
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt * 1000 * step, blit=False)

        if save_gif:
            print(f"Bezig met renderen en opslaan van {filename}...")
            print("(Dit kan een minuutje duren, de grafiek opent nu even niet. Wacht rustig af...)")

            # Gebruik de PillowWriter om een GIF te maken
            writer = animation.PillowWriter(fps=int(1 / (dt * step)))
            ani.save(filename, writer=writer)

            print("✅ GIF succesvol opgeslagen in de map Results!")
        else:
            plt.show()