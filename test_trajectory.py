import numpy as np
from trajectory import TrajectoryGenerator


def test_trajectory():
    print("--- Test: Quintic Polynomial Trajectory ---")

    start = np.array([0.0, 0.0, 0.0])
    doel = np.array([5.0, -2.0, 10.0])
    vliegtijd = 5.0  # seconden

    # Maak de generator aan
    planner = TrajectoryGenerator(start_pos=start, end_pos=doel, T=vliegtijd, start_yaw=0.0, end_yaw=np.pi / 2)

    # Evalueer op verschillende tijdstippen
    check_times = [0.0, 2.5, 5.0, 6.0]  # 6.0 ligt buiten de vliegtijd, drone moet stilstaan op het einddoel

    for t in check_times:
        ref = planner.get_reference(t)
        print(f"\n[ Tijd = {t} sec ]")
        print(f"  Positie : {np.round(ref['pos'], 3)}")
        print(f"  Snelheid: {np.round(ref['vel'], 3)}")
        print(f"  Yaw (rad): {ref['yaw']:.3f}")

    # Validatie: Controleer of we exact op het einddoel zijn op t=5
    ref_end = planner.get_reference(5.0)
    pos_error = np.linalg.norm(ref_end['pos'] - doel)
    vel_error = np.linalg.norm(ref_end['vel'])

    print("\n--- Validatie ---")
    if pos_error < 1e-4 and vel_error < 1e-4:
        print("✅ SUCCES: Het traject brengt de drone gladjes en exact naar het doel!")
    else:
        print("❌ FOUT: De randvoorwaarden kloppen niet.")


if __name__ == "__main__":
    test_trajectory()