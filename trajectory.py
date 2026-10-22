import numpy as np


class QuinticPolynomial:
    """
    Genereert een 5e-orde polynoom voor 1 dimensie (bijv. enkel de X-as).
    Zorgt voor continue positie, snelheid en versnelling.
    """

    def __init__(self, p0: float, v0: float, a0: float,
                 pf: float, vf: float, af: float, T: float):
        """
        p0, v0, a0 : Begin positie, snelheid, versnelling
        pf, vf, af : Eind positie, snelheid, versnelling
        T : Totale tijd voor het traject
        """
        self.T = T

        # De matrix A (6x6) bevat de machten van t voor t=0 en t=T
        # Voor t=0 vallen alle termen met t weg, dus de eerste 3 rijen zijn triviaal:
        # p(0) = c0
        # v(0) = c1
        # a(0) = 2*c2
        A = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, T, T ** 2, T ** 3, T ** 4, T ** 5],
            [0.0, 1.0, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [0.0, 0.0, 2.0, 6 * T, 12 * T ** 2, 20 * T ** 3]
        ])

        # De vector b bevat onze gewenste randvoorwaarden
        b = np.array([p0, v0, a0, pf, vf, af])

        # Los het stelsel A * c = b op voor de coëfficiënten c (c0 t/m c5)
        # We gebruiken np.linalg.solve, dit is numeriek stabieler dan A_inv @ b
        self.c = np.linalg.solve(A, b)

    def evaluate(self, t: float) -> tuple:
        """
        Berekent de positie, snelheid en versnelling op tijdstip t.
        """
        # Klem de tijd tussen 0 en T, zodat hij op z'n einddoel blijft stilstaan
        t = np.clip(t, 0.0, self.T)

        c0, c1, c2, c3, c4, c5 = self.c

        # Evaluatie van p(t), v(t), a(t)
        p = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
        v = c1 + 2 * c2 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
        a = 2 * c2 + 6 * c3 * t + 12 * c4 * t ** 2 + 20 * c5 * t ** 3

        return p, v, a


class TrajectoryGenerator:
    """
    Bundelt meerdere QuinticPolynomials samen voor 3D vliegen (X, Y, Z, Yaw).
    """

    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, T: float, start_yaw: float = 0.0,
                 end_yaw: float = 0.0):
        # We gaan uit van 'rest-to-rest' beweging (snelheid & versnelling = 0 aan begin/eind)
        self.T = T

        self.traj_x = QuinticPolynomial(start_pos[0], 0, 0, end_pos[0], 0, 0, T)
        self.traj_y = QuinticPolynomial(start_pos[1], 0, 0, end_pos[1], 0, 0, T)
        self.traj_z = QuinticPolynomial(start_pos[2], 0, 0, end_pos[2], 0, 0, T)
        self.traj_yaw = QuinticPolynomial(start_yaw, 0, 0, end_yaw, 0, 0, T)

    def get_reference(self, t: float):
        """
        Geeft de volledige referentie-toestand terug voor de regelaar (Controller) op tijdstip t.
        Returns dict: {'pos', 'vel', 'acc', 'yaw', 'yaw_rate'}
        """
        px, vx, ax = self.traj_x.evaluate(t)
        py, vy, ay = self.traj_y.evaluate(t)
        pz, vz, az = self.traj_z.evaluate(t)
        pyaw, vyaw, _ = self.traj_yaw.evaluate(t)

        return {
            'pos': np.array([px, py, pz]),
            'vel': np.array([vx, vy, vz]),
            'acc': np.array([ax, ay, az]),
            'yaw': pyaw,
            'yaw_rate': vyaw
        }
