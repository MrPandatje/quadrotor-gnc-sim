from config import default_config


def run_sanity_check():
    print("--- Quadrotor Configuratie Test ---")
    print(f"Massa: {default_config.mass} kg")
    print(f"Traagheidsmatrix J:\n{default_config.J}")

    # Kleine wiskundige sanity check: J * J_inv moet een Identiteitsmatrix opleveren
    identity_test = default_config.J @ default_config.J_inv

    import numpy as np
    # np.allclose checkt of matrices gelijk zijn, rekening houdend met afrondingsfouten
    is_valid = np.allclose(identity_test, np.eye(3))

    print(f"\nJ * J_inv is Identiteitsmatrix? {'JA ✅' if is_valid else 'NEE ❌'}")


if __name__ == "__main__":
    run_sanity_check()