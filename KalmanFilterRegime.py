import numpy as np
from scipy.linalg import solve_discrete_are


class KalmanFilterRegime():
    def __init__(self, A, H, Q, R, X):
        """
        Kalman Filter implementation with no control inputs (B, u).

        Inputs:
            A (np.array) = state transition matrix of the process from state at time k to the state at time k+1 (sometimes denoted as Phi)
            H (np.array) = (measurement matrix) noiseless connection btw the state vector and the measurement vector
            Q (np.array) = process noise covariance matrix
            R (np.array) = measurement noise covariance matrix
            X (np.array) = initial state vector

        """
        self.A = A # state transition matrix of the process from state at time k to the state at time k+1 (sometimes denoted as Phi)
        self.H = H # (measurement matrix) noiseless connection btw the state vector and the measurement vector
        self.Q = Q # process noise covariance matrix
        self.R = R # measurement noise covariance matrix
        self.X = X # state vector

        # Steady-state: risolve la DARE e calcola K_inf una volta sola
        self.P_inf = solve_discrete_are(self.A.T, self.H.T, self.Q, self.R)
        self.K_inf = self.P_inf @ self.H.T @ np.linalg.inv(self.H @ self.P_inf @ self.H.T + self.R)

    def predict(self):
        """
        Performs the predict step in the Kalman filter which simply propagates the previous state 
        estimate forward using the system's state transition model, while also updating the 
        uncertainty based on the process noise covariance.

        Returns:
            self.X: state vector estimate obtained by the prediction step.
        """
        # State Projection:
        # X_t = A X_{t-1}
        self.X = np.matmul(self.A, self.X)
        return self.X

    def update(self, Y):
        """
        The update step in the Kalman filter corrects the predicted state estimate by 
        incorporating new measurements, adjusting the estimate and its uncertainty based on 
        the measurement residual and the Kalman gain (K).

        Inputs:
            Y (np.array): measurement vector
        Returns:
            self.X (np.array): updated state vector estimate obtained after the update step.
        """
  
        self.X = self.X + np.matmul(self.K_inf, (Y - np.matmul(self.H, self.X)))

        return self.X