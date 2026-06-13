import numpy as np


class UKF():
    def __init__(self, n, m, f, h, Q, R, a=1e-3, b=2, k=0):
        #coefficienti per il calcolo dei pesi self.a, self.b, gamma, k e la dimensione dello stato
        self.a = a
        self.b = b
        self.k = k
        self.n = n
        self.m = m
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

        self.lambda_ = a**2 * (n + k) - n

        self.Wm, self.Wc = self.compute_weights()        
        
        #definisci lo stato iniziale
        self.X = np.zeros(n)
        self.P = np.eye(n)

    def compute_weights(self):
        wm = [0.0] * (2 * self.n + 1)
        wc = [0.0] * (2 * self.n + 1)

        wm[0] = self.lambda_ / (self.n + self.lambda_)
        wc[0] = wm[0] + (1.0 - self.a**2 + self.b)

        w = 1.0 / (2.0 * (self.n + self.lambda_))

        for i in range(1, 2 * self.n + 1):
            wm[i] = w
            wc[i] = w

        return wm, wc

        
    def gen_sigma(self):
        #già so che sono 11x5 quindi mi predispongo
        L = np.linalg.cholesky((self.n + self.lambda_) * self.P)
        sigma_points = np.zeros((self.n, 2*self.n+1))
        sigma_points[:, 0] = self.X                    # punto centrale
        sigma_points[:, 1:self.n+1] = self.X[:, None] + L   # +L_i colonne 1..5
        sigma_points[:, self.n+1:]  = self.X[:, None] - L   # -L_i colonne 6..10

        return sigma_points

    def predict(self, dt):
        n = self.n
        sigma = self.gen_sigma()

        # propagate sigma points through f
        sigma_pred = np.zeros_like(sigma)
        for i in range(2*n+1):
            sigma_pred[:, i] = self.f(sigma[:, i], dt)

        # predicted mean
        x_pred = sigma_pred @ self.Wm

        # predicted covariance
        diff = sigma_pred - x_pred[:, None]
        P_pred = (diff * self.Wc) @ diff.T + self.Q

        self.X = x_pred
        self.P = P_pred
        self.sigma_pred = sigma_pred  # store for update step

    def update(self, y):

        n, m = self.n, self.m

        # propagate sigma points through h
        sigma_obs = np.zeros((m, 2*n+1))
        for i in range(2*n+1):
            sigma_obs[:, i] = self.h(self.sigma_pred[:, i])

        # predicted observation mean
        y_pred = sigma_obs @ self.Wm

        # innovation covariance S
        diff_y = sigma_obs - y_pred[:, None]
        S = (diff_y * self.Wc) @ diff_y.T + self.R

        # cross covariance T
        diff_x = self.sigma_pred - self.X[:, None]
        T = (diff_x * self.Wc) @ diff_y.T

        # Kalman gain
        K = T @ np.linalg.inv(S)

        # update state and covariance
        self.X = self.X + K @ (y - y_pred)
        self.P = self.P - K @ S @ K.T

        return self.X, self.P