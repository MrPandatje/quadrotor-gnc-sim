import numpy as np
from config import QuadrotorConfig


class CascadedController:
    def __init__(self, config: QuadrotorConfig):
        self.config = config

        # Buitenlus Gains (Positie & Snelheid)
        self.Kp_pos = np.array([5.0, 5.0, 10.0])  # Z heeft vaak meer sturing nodig
        self.Kd_pos = np.array([4.0, 4.0, 6.0])

        # Binnenlus Gains (Attitude & Hoeksnelheid)
        self.Kp_att = np.array([200.0, 200.0, 100.0])
        self.Kd_att = np.array([40.0, 40.0, 30.0])

    def get_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        # Kopie uit drone_dynamics voor lokaal gebruik
        qw, qx, qy, qz = q / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

    def compute_control(self, state: np.ndarray, reference: dict) -> np.ndarray:
        """
        Berekent de 4 motorsnelheden op basis van de huidige state en de referentie (doel).
        """
        # Huidige toestand uitpakken
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        R = self.get_rotation_matrix(q)
        z_body = R[:, 2]  # De huidige Z-as van de drone (waar de neus/bovenkant heen wijst)

        # --- 1. Buitenlus: Positieregeling ---
        ep = reference['pos'] - pos
        ev = reference['vel'] - vel

        # Gewenste versnelling (a_des = a_ref + Kp*ep + Kd*ev)
        a_des = reference['acc'] + self.Kp_pos * ep + self.Kd_pos * ev

        # Gewenste krachtvector (Voeg zwaartekracht-compensatie toe in de Z-richting)
        gravity_vec = np.array([0, 0, self.config.g])
        F_des = self.config.mass * (a_des + gravity_vec)

        # Totale stuwkracht (T) is de projectie van F_des op de body Z-as
        T = np.dot(F_des, z_body)

        # --- 2. Attitude Extractie (Vind gewenste rotatie) ---
        # Normalizeer F_des om de gewenste Z-as te vinden
        z_des = F_des / (np.linalg.norm(F_des) + 1e-6)  # Voorkom deling door 0

        # Gewenste X-as (bepaald door de gewenste Yaw hoek)
        yaw_ref = reference['yaw']
        x_c = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])

        # Gewenste Y-as staat loodrecht op z_des en x_c
        y_des = np.cross(z_des, x_c)
        y_des = y_des / np.linalg.norm(y_des)

        # Her-bereken gewenste X-as zodat alles perfect haaks op elkaar staat
        x_des = np.cross(y_des, z_des)

        # Gewenste Rotatiematrix
        R_des = np.column_stack((x_des, y_des, z_des))

        # --- 3. Binnenlus: Attitude Regeling ---
        # Bereken de rotatiefout: e_R = 0.5 * (R_des^T * R - R^T * R_des)_vee
        # Dit is de formele geometrische manier om hoekfouten in SO(3) te berekenen.
        e_R_matrix = 0.5 * (R.T @ R_des - R_des.T @ R)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])  # "vee" map

        # We negeren referentie hoeksnelheid voor eenvoud in deze stap, dus e_omega = -omega
        e_omega = -omega

        # Bereken benodigde koppels (Torques)
        tau = self.Kp_att * e_R + self.Kd_att * e_omega
        # Fysisch gezien moeten we nog vermenigvuldigen met traagheidsmatrix J
        tau = self.config.J @ tau

        # --- 4. Motor Allocatie (De Mixer) ---
        # We lossen A * x = b op, waarbij x = [w1^2, w2^2, w3^2, w4^2]
        kf, km, L = self.config.kf, self.config.km, self.config.L
        l = L * (np.sqrt(2) / 2)

        # Gecorrigeerde Inverse Allocation Matrix (Exact in sync met dynamica)
        A = np.array([
            [kf, kf, kf, kf],
            [kf * l, -kf * l, kf * l, -kf * l],
            [-kf * l, kf * l, kf * l, -kf * l],
            [km, km, -km, -km]
        ])

        b = np.array([T, tau[0], tau[1], tau[2]])

        # Bereken de kwadraten van de motorsnelheden
        # np.linalg.solve crasht niet op singuliere matrices als de drone config goed is.
        w_sq = np.linalg.solve(A, b)

        # Bescherming: Een motor kan niet 'negatief' draaien om stuwkracht te leveren.
        w_sq = np.maximum(w_sq, 0.0)

        # Neem de wortel voor de uiteindelijke rad/s
        motor_speeds = np.sqrt(w_sq)

        return motor_speeds