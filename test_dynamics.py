import numpy as np
from config import default_config
from drone_dynamics import QuadrotorDynamics


def test_hover():
    print("--- Test: Hovering Dynamics ---")
    drone = QuadrotorDynamics(default_config)

    # Bereken de motorsnelheid nodig om exact stil te hangen
    # 4 * kf * w^2 = m * g  => w = sqrt((m*g) / (4*kf))
    hover_omega = np.sqrt((default_config.mass * default_config.g) / (4 * default_config.kf))
    print(f"Berekende zweef-snelheid per motor: {hover_omega:.2f} rad/s")

    motor_speeds = np.array([hover_omega, hover_omega, hover_omega, hover_omega])

    # Simuleer 100 stappen van 0.01 seconde (totaal 1 seconde)
    dt = 0.01
    for _ in range(100):
        drone.step(dt, motor_speeds)

    z_position = drone.state[2]
    z_velocity = drone.state[5]

    print(f"Na 1 seconde simulatie:")
    print(f"Hoogte (Z): {z_position:.6f} m")
    print(f"Verticale snelheid (Vz): {z_velocity:.6f} m/s")

    # Check of hij daadwerkelijk is blijven hangen
    if abs(z_position) < 1e-4 and abs(z_velocity) < 1e-4:
        print("✅ SUCCES: De drone kan perfect stil zweven!")
    else:
        print("❌ FOUT: De drone drift weg. Kijk de formules na.")


if __name__ == "__main__":
    test_hover()