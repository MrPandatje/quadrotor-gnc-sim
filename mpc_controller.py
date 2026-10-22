import casadi as ca
import numpy as np
from config import QuadrotorConfig


class NonlinearMPC:
    def __init__(self, config: QuadrotorConfig, N: int = 10, dt: float = 0.1):
        """
        N: Horizon lengte (aantal voorspellingsstappen)
        dt: Tijdstap voor de voorspelling
        """
        self.config = config
        self.N = N
        self.dt = dt

        # Setup CasADi Opti stack (gebruiksvriendelijke interface voor optimalisatie)
        self.opti = ca.Opti()

        # 1. Definieer de symbolische variabelen over de horizon
        # X: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r] (13 variabelen, N+1 stappen)
        self.X = self.opti.variable(13, N + 1)
        # U: [w1^2, w2^2, w3^2, w4^2] (4 variabelen, N stappen).
        # We optimaliseren w^2 i.p.v. w, dit is wiskundig veel stabieler/lineairder voor stuwkracht.
        self.U = self.opti.variable(4, N)

        # Parameters (Inputs die we elke iteratie veranderen)
        self.X0 = self.opti.parameter(13)  # Huidige toestand
        self.X_ref = self.opti.parameter(13)  # Doel toestand

        # 2. Definieer het CasADi Dynamica Model (Continue tijd, X_dot = f(X, U))
        self.f = self._build_dynamics_model()

        # 3. Kostenfunctie (Cost Function) opbouwen
        self._build_cost_function()

        # 4. Randvoorwaarden (Constraints) toevoegen
        self._build_constraints()

        # 5. Solver instellen (IPOPT)
        # Print level 0 verbergt de honderden regels wiskundige output per stap
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100, "print_level": 0, "tol": 1e-4}
        self.opti.solver("ipopt", p_opts, s_opts)

    def _build_dynamics_model(self):
        """Bouwt de symbolische Newton-Euler dynamica op voor CasADi."""
        x = ca.MX.sym('x', 13)
        u = ca.MX.sym('u', 4)  # u is w^2

        # Unpack
        v = x[3:6]
        q = x[6:10]
        omega = x[10:13]

        # Krachten en koppels (Met de gecorrigeerde matrix uit Fase 3!)
        kf, km, L = self.config.kf, self.config.km, self.config.L
        l = L * (np.sqrt(2) / 2)

        T = kf * (u[0] + u[1] + u[2] + u[3])
        tau_x = kf * l * (u[0] - u[1] + u[2] - u[3])
        tau_y = kf * l * (-u[0] + u[1] + u[2] - u[3])
        tau_z = km * (u[0] + u[1] - u[2] - u[3])
        tau = ca.vertcat(tau_x, tau_y, tau_z)

        # Rotatiematrix R(q)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = ca.vertcat(
            ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2))
        )

        # Versnelling
        gravity = ca.vertcat(0, 0, -self.config.g)
        thrust_body = ca.vertcat(0, 0, T)
        acc = gravity + ca.mtimes(R, thrust_body) / self.config.mass

        # Euler rotatie (Omega_dot)
        J = self.config.J
        J_inv = self.config.J_inv
        # J_inv @ (tau - cross(omega, J @ omega))
        gyro = ca.cross(omega, ca.mtimes(J, omega))
        omega_dot = ca.mtimes(J_inv, (tau - gyro))

        # Quaternion kinematica
        p, q_rate, r = omega[0], omega[1], omega[2]
        q_dot = 0.5 * ca.vertcat(
            -qx * p - qy * q_rate - qz * r,
            qw * p + qy * r - qz * q_rate,
            qw * q_rate - qx * r + qz * p,
            qw * r + qx * q_rate - qy * p
        )

        # Combineer tot state_dot
        x_dot = ca.vertcat(v, acc, q_dot, omega_dot)

        # Return als een CasADi functie
        return ca.Function('f', [x, u], [x_dot])

    def _build_cost_function(self):
        """Definieert de strafpunten voor afwijkingen van het doel."""
        cost = 0

        # Gewichtsmatrices (Tuning!)
        # Omdat U (w^2) in de tienduizenden loopt, moet de R straf extreem klein zijn.
        Q_pos = ca.diag([1000.0, 1000.0, 1000.0])  # Zeer hoge drang naar het doel
        Q_vel = ca.diag([10.0, 10.0, 10.0])
        R_ctrl = ca.diag([1e-6, 1e-6, 1e-6, 1e-6])  # 0.000001 (Bijna gratis gas geven)

        for k in range(self.N):
            pos_error = self.X[0:3, k] - self.X_ref[0:3]
            vel_error = self.X[3:6, k] - self.X_ref[3:6]

            # Voeg toe aan totale kosten
            cost += ca.mtimes([pos_error.T, Q_pos, pos_error])
            cost += ca.mtimes([vel_error.T, Q_vel, vel_error])

            # In hover, w^2 = m*g/(4*kf). We bestraffen de afwijking daarvan.
            w_sq_hover = (self.config.mass * self.config.g) / (4 * self.config.kf)
            u_error = self.U[:, k] - w_sq_hover
            cost += ca.mtimes([u_error.T, R_ctrl, u_error])

        self.opti.minimize(cost)

    def _build_constraints(self):
        """Fysieke limieten (Dynamica, actuatoren)."""
        # Beginvoorwaarde (We moeten starten waar de drone NU is)
        self.opti.subject_to(self.X[:, 0] == self.X0)

        # Meerdere schieten (Multiple Shooting): Koppel de stappen via RK4 integratie
        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]

            # RK4 in CasADi
            k1 = self.f(x_k, u_k)
            k2 = self.f(x_k + 0.5 * self.dt * k1, u_k)
            k3 = self.f(x_k + 0.5 * self.dt * k2, u_k)
            k4 = self.f(x_k + self.dt * k3, u_k)
            x_next = x_k + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Dwing de optimizer om de fysische predictie te volgen
            self.opti.subject_to(self.X[:, k + 1] == x_next)

            # Motor limieten (w^2 moet tussen 0 en max^2 liggen)
            w_sq_max = self.config.rotor_speed_max ** 2
            self.opti.subject_to(self.opti.bounded(0, u_k, w_sq_max))

    def compute_control(self, state: np.ndarray, state_ref: np.ndarray) -> np.ndarray:
        """Lost het probleem op en geeft de optimale motorsnelheden voor de HUIDIGE stap."""
        # Update de parameters in de solver
        self.opti.set_value(self.X0, state)
        self.opti.set_value(self.X_ref, state_ref)

        # Optioneel: Initial guess meegeven helpt de solver (gebruik vorige oplossing)
        # Voor deze fase houden we het even simpel zonder warm-starts

        try:
            sol = self.opti.solve()
            # Haal de optimale U van de EERSTE stap uit de horizon (Receding Horizon principe)
            u_opt_sq = sol.value(self.U[:, 0])
            return np.sqrt(np.maximum(u_opt_sq, 0.0))
        except Exception as e:
            # Als de solver faalt (infeasible), hover dan als noodoplossing
            print(f"MPC Fout/Infeasible. Nood-hover geactiveerd.")
            hover_w = np.sqrt((self.config.mass * self.config.g) / (4 * self.config.kf))
            return np.array([hover_w, hover_w, hover_w, hover_w])