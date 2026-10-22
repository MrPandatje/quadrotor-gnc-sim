import numpy as np


class SensorSimulator:
    def __init__(self, dt_imu: float = 0.01, dt_gps: float = 0.1):
        self.dt_imu = dt_imu
        self.dt_gps = dt_gps

        # Sensor Ruis Parameters (Standaardafwijkingen - Sigma)
        self.sigma_acc = 0.1  # m/s^2 ruis op versnellingsmeter
        self.sigma_gyro = 0.01  # rad/s ruis op gyroscoop
        self.sigma_gps = 0.5  # 0.5 meter onnauwkeurigheid op GPS

        self.time_since_last_gps = 0.0

    def get_imu_measurement(self, true_acc: np.ndarray, true_omega: np.ndarray):
        """Voegt Gaussiaanse (witte) ruis toe aan de ware IMU waarden."""
        acc_noisy = true_acc + np.random.normal(0, self.sigma_acc, 3)
        omega_noisy = true_omega + np.random.normal(0, self.sigma_gyro, 3)
        return acc_noisy, omega_noisy

    def get_gps_measurement(self, true_pos: np.ndarray, dt: float):
        """Geeft alleen een GPS meting als het tijd is (GPS is trager dan IMU)."""
        self.time_since_last_gps += dt
        if self.time_since_last_gps >= self.dt_gps - 1e-5:
            self.time_since_last_gps = 0.0
            pos_noisy = true_pos + np.random.normal(0, self.sigma_gps, 3)
            return True, pos_noisy
        return False, np.zeros(3)