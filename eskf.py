import numpy as np


class ESKF:
    def __init__(self):
        # 1. Nominale Toestand: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.nom_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # 2. Fout Covariantiematrix P (9x9)
        # (Onzekerheid in: [Positie_xyz, Snelheid_xyz, Hoek_xyz])
        self.P = np.eye(9) * 0.1

        # 3. Proces Ruis Q (IMU onzekerheid)
        self.Q = np.eye(9)
        self.Q[0:3, 0:3] *= 0.0
        self.Q[3:6, 3:6] *= 0.1  # Meer wantrouwen in eigen snelheid
        self.Q[6:9, 6:9] *= 0.01  # Meer wantrouwen in eigen gyroscoop

        # 4. Meet Ruis R (GPS onzekerheid)
        self.R_gps = np.eye(3) * 0.5 ** 2  # Sigma = 0.5m

        self.g = np.array([0, 0, -9.81])

    def _get_R(self, q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Maakt een scheef-symmetrische matrix voor kruisproducten [v]_x"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict_imu(self, acc_m: np.ndarray, gyro_m: np.ndarray, dt: float):
        """Stap 1: Propagatie van de nominale toestand en onzekerheid met IMU."""
        p = self.nom_state[0:3]
        v = self.nom_state[3:6]
        q = self.nom_state[6:10]

        R = self._get_R(q)

        # --- Nominale state update ---
        acc_world = R @ acc_m + self.g

        # Euler integratie (voor filter vaak goed genoeg mits dt klein is)
        p_new = p + v * dt + 0.5 * acc_world * dt ** 2
        v_new = v + acc_world * dt

        # Quaternion update (met kleine hoekbenadering)
        omega_mag = np.linalg.norm(gyro_m)
        if omega_mag > 1e-6:
            dq = np.array([np.cos(omega_mag * dt / 2)] + list(np.sin(omega_mag * dt / 2) * (gyro_m / omega_mag)))
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])

        # Hamilton product: q_new = q * dq
        qw, qx, qy, qz = q
        dw, dx, dy, dz = dq
        q_new = np.array([
            qw * dw - qx * dx - qy * dy - qz * dz,
            qw * dx + qx * dw + qy * dz - qz * dy,
            qw * dy - qx * dz + qy * dw + qz * dx,
            qw * dz + qx * dy - qy * dx + qz * dw
        ])
        q_new = q_new / np.linalg.norm(q_new)

        self.nom_state[0:3] = p_new
        self.nom_state[3:6] = v_new
        self.nom_state[6:10] = q_new

        # --- Fout Covariantie update (De wiskunde van onzekerheid) ---
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        # Jacobiaan van versnelling t.o.v. rotatie (F_v_theta = -R * [acc]_x * dt)
        F[3:6, 6:9] = -R @ self._skew_symmetric(acc_m) * dt

        # P = F * P * F^T + Q * dt (Standaard Random Walk iteratie)
        self.P = F @ self.P @ F.T + (self.Q * dt)

    def update_gps(self, gps_pos: np.ndarray):
        """Stap 2: GPS meting komt binnen, corrigeer de voorspelling!"""
        # H is de observatiematrix (GPS meet alleen de eerste 3 states: Positie)
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)

        # Rest (verschil tussen GPS en onze schatting)
        y = gps_pos - self.nom_state[0:3]

        # S: Innovatie Covariantie
        S = H @ self.P @ H.T + self.R_gps

        # K: Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Fout-Toestand (\delta x = K * y)
        delta_x = K @ y

        # --- Stap 3: Fout Injecteren in Nominale Toestand ---
        self.nom_state[0:3] += delta_x[0:3]  # Positie correctie
        self.nom_state[3:6] += delta_x[3:6]  # Snelheid correctie

        # Attitude correctie (Hoeken naar Quaternion omzetten)
        dtheta = delta_x[6:9]
        dtheta_mag = np.linalg.norm(dtheta)
        if dtheta_mag > 1e-6:
            dq_err = np.array([np.cos(dtheta_mag / 2)] + list(np.sin(dtheta_mag / 2) * (dtheta / dtheta_mag)))
        else:
            dq_err = np.array([1.0, 0.0, 0.0, 0.0])

        qw, qx, qy, qz = self.nom_state[6:10]  # Dit is q_nom
        dw, dx, dy, dz = dq_err  # Dit is de fout

        # CORRECTE Hamilton product: q_nom * dq_err (Lokale Fout Injectie)
        q_corr = np.array([
            qw * dw - qx * dx - qy * dy - qz * dz,
            qw * dx + qx * dw + qy * dz - qz * dy,
            qw * dy - qx * dz + qy * dw + qz * dx,
            qw * dz + qx * dy - qy * dx + qz * dw
        ])
        self.nom_state[6:10] = q_corr / np.linalg.norm(q_corr)

        # --- Stap 4: Reset Covariantie P = (I - K*H) * P ---
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P