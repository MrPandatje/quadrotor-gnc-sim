import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuadrotorConfig:
    """
    Configuratieparameters voor de quadrotor.
    'frozen=True' zorgt ervoor dat we deze waarden tijdens de simulatie
    niet per ongeluk kunnen overschrijven.
    """
    # Zwaartekracht (m/s^2)
    g: float = 9.81

    # Massa van de drone (kg)
    mass: float = 1.2

    # Traagheidsmomenten (Inertia) (kg*m^2)
    # We slaan dit direct op als een 3x3 diagonale numpy matrix
    Ixx: float = 0.0123
    Iyy: float = 0.0123
    Izz: float = 0.0224
    J: np.ndarray = field(init=False)
    J_inv: np.ndarray = field(init=False)

    # Geometrie
    # Armlengte: afstand van zwaartepunt (CoG) tot de motor (m)
    L: float = 0.25

    # Aerodynamische constanten
    # Stuwkrachtconstante (Thrust coefficient) F = kf * w^2
    kf: float = 5.0e-5
    # Koppelconstante (Torque/Moment coefficient) M = km * w^2
    km: float = 1.5e-6

    # Motor limieten (rad/s)
    rotor_speed_min: float = 0.0
    rotor_speed_max: float = 800.0

    def __post_init__(self):
        """
        Deze methode wordt automatisch gerund na de initialisatie.
        Hier berekenen we afgeleide variabelen zoals de traagheidsmatrix.
        We gebruiken object.__setattr__ omdat de dataclass 'frozen' is.
        """
        # Creëer de diagonale traagheidsmatrix J
        inertia_matrix = np.array([
            [self.Ixx, 0.0, 0.0],
            [0.0, self.Iyy, 0.0],
            [0.0, 0.0, self.Izz]
        ], dtype=float)
        object.__setattr__(self, 'J', inertia_matrix)

        # Bereken direct de inverse (veel nodig in dynamica berekeningen)
        object.__setattr__(self, 'J_inv', np.linalg.inv(inertia_matrix))


# Maak een standaard configuratie object aan dat we doorheen
# het project kunnen importeren
default_config = QuadrotorConfig()