import numpy as np
from config import QuadrotorConfig


class QuadrotorDynamics:
    def __init__(self, config: QuadrotorConfig):
        self.config = config

        # Initialiseer de toestand (State: x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r)
        # We starten in de oorsprong, in rust, perfect horizontaal (qw=1, rest=0)
        self.state = np.array([
            0.0, 0.0, 0.0,  # Positie: x, y, z
            0.0, 0.0, 0.0,  # Snelheid: vx, vy, vz
            1.0, 0.0, 0.0, 0.0,  # Quaternion: qw, qx, qy, qz (Hamilton)
            0.0, 0.0, 0.0  # Hoeksnelheid: p, q, r
        ], dtype=float)

    def get_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Zet een Hamilton quaternion [qw, qx, qy, qz] om naar een 3x3 Rotatiematrix (Body to World).
        """
        qw, qx, qy, qz = q

        # Voorkom wiskundige fouten door de quaternion te normaliseren
        norm = np.linalg.norm(q)
        if norm > 0:
            qw, qx, qy, qz = q / norm

        # Formule voor rotatiematrix vanuit quaternion
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])
        return R

    def calculate_forces_and_torques(self, motor_speeds: np.ndarray):
        """
        Zet 4 motorsnelheden (rad/s) om naar Thrust (T) en Torques (tau_x, tau_y, tau_z).
        Gaat uit van een 'X' configuratie in een ENU frame (X=Rechts, Y=Voor, Z=Boven).
        """
        w_sq = np.square(motor_speeds)  # Omega^2
        kf = self.config.kf
        km = self.config.km
        L = self.config.L

        # Totale stuwkracht (wijst in de Body +Z richting)
        T = kf * np.sum(w_sq)

        # Geometrische factor voor X-configuratie (arm in 45 graden hoek met assen)
        # Afstand van as tot motor is L * sin(45) = L * (sqrt(2)/2)
        l = L * (np.sqrt(2) / 2)

        # Allocation Matrix
        # M1 (Rechts-Voor, CW), M2 (Links-Achter, CW), M3 (Links-Voor, CCW), M4 (Rechts-Achter, CCW)
        tau_x = kf * l * (-w_sq[0] - w_sq[1] + w_sq[2] + w_sq[3])  # Roll (draaien om X)
        tau_y = kf * l * (-w_sq[0] + w_sq[1] + w_sq[2] - w_sq[3])  # Pitch (draaien om Y)
        tau_z = km * (-w_sq[0] - w_sq[1] + w_sq[2] + w_sq[3])  # Yaw (draaien om Z, gebaseerd op motor draairichting)

        return T, np.array([tau_x, tau_y, tau_z])

    def calculate_derivatives(self, state: np.ndarray, motor_speeds: np.ndarray) -> np.ndarray:
        """
        Berekent de afgeleide van de toestand (X_dot = f(X, U)) voor de RK4 integratie.
        """
        # Unpack de state
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        # Berekent krachten en koppels vanuit actuatormodel
        T, tau = self.calculate_forces_and_torques(motor_speeds)

        # 1. Translatiemodel (Newton)
        R = self.get_rotation_matrix(q)
        gravity = np.array([0.0, 0.0, -self.config.g])
        thrust_body = np.array([0.0, 0.0, T])
        acceleration = gravity + (R @ thrust_body) / self.config.mass

        # 2. Rotatiemodel (Euler)
        # J_inv * (tau - (omega x (J * omega)))
        gyroscopic_effect = np.cross(omega, self.config.J @ omega)
        omega_dot = self.config.J_inv @ (tau - gyroscopic_effect)

        # 3. Quaternion Kinematica
        # q_dot = 0.5 * q * [0, p, q, r] (Hamilton vermenigvuldiging)
        qw, qx, qy, qz = q
        p, q_rate, r = omega  # 'q_rate' om verwarring met quaternion 'q' te voorkomen

        q_dot = 0.5 * np.array([
            -qx * p - qy * q_rate - qz * r,
            qw * p + qy * r - qz * q_rate,
            qw * q_rate - qx * r + qz * p,
            qw * r + qx * q_rate - qy * p
        ])

        # Pak alles samen in één afgeleide vector
        state_dot = np.zeros(13)
        state_dot[0:3] = v  # p_dot = v
        state_dot[3:6] = acceleration
        state_dot[6:10] = q_dot
        state_dot[10:13] = omega_dot

        return state_dot

    def step(self, dt: float, motor_speeds: np.ndarray):
        """
        RK4 (Runge-Kutta 4e orde) Integratie om de simulatie 1 tijdstap (dt) vooruit te zetten.
        Dit is veel nauwkeuriger dan simpele Euler integratie.
        """
        # Klem de motorsnelheden tussen min en max uit de config
        motor_speeds = np.clip(motor_speeds, self.config.rotor_speed_min, self.config.rotor_speed_max)

        k1 = self.calculate_derivatives(self.state, motor_speeds)
        k2 = self.calculate_derivatives(self.state + 0.5 * dt * k1, motor_speeds)
        k3 = self.calculate_derivatives(self.state + 0.5 * dt * k2, motor_speeds)
        k4 = self.calculate_derivatives(self.state + dt * k3, motor_speeds)

        self.state = self.state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Na integratie MOETEN we de quaternion weer normaliseren (lengte = 1),
        # anders veroorzaakt numerieke fout opbouw dat de rotatiematrix vervormt.
        q = self.state[6:10]
        self.state[6:10] = q / np.linalg.norm(q)