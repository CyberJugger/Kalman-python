# 2D Mouse Tracking with Kalman Filter and UKF

A pygame-based interactive demo that tracks mouse movement in real time using two different filtering approaches: a classical **Kalman Filter** with constant acceleration kinematics, and an **Unscented Kalman Filter (UKF)** with a nonlinear **CTRV** (Constant Turn Rate and Velocity) motion model.

---

## Motivation

Raw mouse input is noisy. A naive approach would use the measurements directly, but this gives jittery trajectories. Both filters maintain a belief over the mouse state (position, velocity, acceleration or turn rate) and produce a smoothed, physically consistent estimate by fusing the kinematic model with noisy measurements.

The two filters represent different trade-offs:

- **Kalman Filter** — linear model, fast, works well for straight or slowly curving motion
- **UKF + CTRV** — nonlinear model, handles sharp turns and curved trajectories more accurately at the cost of slightly higher compute
- **Steady-State Kalman Filter** — same linear model as the KF, but with the Kalman gain pre-computed offline; zero per-step covariance update

---

## Project Structure

```
project/
│
├── models/
│   └── ctrv.py                  # CTRV transition and observation functions
│
├── KalmanFilter.py             # Classical Kalman Filter class 
├── KalmanFilterRegime.py       # Steady-state Kalman Filter class
├── UnscentedKalmanFilter.py   # UKF class
│
├── interactive_kf.py            # Pygame demo — Kalman Filter
├── interactive_rkf.py         # Pygame demo — Steady-State Kalman Filter
└── interactive_ukf.py           # Pygame demo — UKF with CTRV
```

---

## Motion Models

### Kalman Filter — Constant Acceleration (CA)

State vector: $[x,\ y,\ \dot{x},\ \dot{y},\ \ddot{x},\ \ddot{y}]^T$

The model assumes constant acceleration with jerk treated as process noise. The transition matrix A integrates kinematics over one time step $\Delta t$. Q is derived analytically from the jerk variance $\sigma_j^2$:

$$Q = \sigma_j^2 \begin{bmatrix} \frac{\Delta t^6}{36} & \cdots \\ \vdots & \ddots \end{bmatrix}$$

Measurement: position $[x,\ y]^T$ only.

### Steady-State Kalman Filter — Constant Acceleration (CA)

Same CA model as above. Because the system is **Linear Time-Invariant** (A, H, Q, R are all constant for fixed `fps`), the error covariance P converges to a stationary value P∞ and the Kalman gain converges to a constant K∞. These are computed once at startup by solving the **Discrete Algebraic Riccati Equation (DARE)**:

$$P_\infty = A P_\infty A^T - A P_\infty H^T (H P_\infty H^T + R)^{-1} H P_\infty A^T + Q$$

The stationary gain is then:

$$K_\infty = P_\infty H^T (H P_\infty H^T + R)^{-1}$$

At runtime the update step reduces to a single matrix-vector multiply, with no covariance propagation:

$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_\infty (z_k - H \hat{x}_{k|k-1})$$

The initial covariance is set to P∞ directly (`kf.P = kf.P_inf`), so the filter starts at regime with no transient.

### UKF — CTRV (Constant Turn Rate and Velocity)

State vector: $[p_x,\ p_y,\ v,\ \theta,\ \omega]^T$

The nonlinear transition integrates the CTRV equations exactly over $\Delta t$:

$$p_x(k+1) = p_x(k) + \frac{v}{\omega}\bigl(\sin(\theta + \omega\Delta t) - \sin\theta\bigr)$$

$$p_y(k+1) = p_y(k) + \frac{v}{\omega}\bigl(-\cos(\theta + \omega\Delta t) + \cos\theta\bigr)$$

with a fallback to straight-line motion when $\omega \approx 0$.

Process noise enters on $v$ and $\omega$ (unmodeled acceleration and turn rate change). Measurement: position $[p_x,\ p_y]^T$ from the mouse.

The UKF propagates $2n+1 = 11$ sigma points through the exact nonlinear model — no Jacobians, no linearization.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Run the classical Kalman Filter demo:

```bash
python interactive_kf.py
```

Run the Steady-State Kalman Filter demo:

```bash
python interactive_rkf.py
```

Run the UKF with CTRV demo:

```bash
python interactive_ukf.py
```

Move the mouse around the window. The black line follows the raw mouse input; the red line shows the filter estimate. In the UKF demo, blue dots show the 11 sigma points — their spread reflects the current positional uncertainty.

---

## Parameters

### Kalman Filter (`interactive_kf.py`) and Steady-State KF (`interactive_rkf.py`)

| Parameter | Description | Default |
|---|---|---|
| `fps` | Frame rate | `60` |
| `sj2` | Jerk variance (process noise) | `1800` |
| `R` | Measurement noise covariance | `diag(1, 1)` |

### UKF (`interactive_ukf.py`)

| Parameter | Description | Default |
|---|---|---|
| `fps` | Frame rate | `60` |
| `sigma_a` | Std of linear acceleration noise | `1.0` |
| `sigma_omega` | Std of angular acceleration noise | `0.1` |
| `sigma_gps` | Std of position measurement noise | `2.0` |

**Tuning intuition:**

- increase `sigma_a` / `sigma_omega` → filter trusts the model less, follows the mouse more reactively
- increase `sigma_gps` → filter trusts measurements less, produces a smoother but laggier trace
- the ratio between process and measurement noise is what matters, not the absolute values

---

## Visualization

| Color | Meaning |
|---|---|
| Black | Raw mouse position |
| Red | Filter estimate (KF or UKF) |
| Blue dots | UKF sigma points (uncertainty cloud) |

---

## Theory Notes

Both filters share the same predict → update cycle:

1. **Predict** — propagate the state estimate and covariance forward using the motion model
2. **Update** — correct the prediction using the new mouse position measurement

The Kalman Filter does this analytically (linear algebra). The UKF does this by sampling 11 deterministic sigma points around the current estimate, propagating each through the exact nonlinear model, and reconstructing mean and covariance from the propagated cloud — capturing up to third-order accuracy in the Taylor expansion of the nonlinearity.

The Steady-State Kalman Filter skips the covariance update entirely at runtime. Since the CA model is LTI, the DARE has a unique positive-definite solution P∞ (guaranteed by observability of (A, H) and stabilizability of (A, Q^{1/2})). The resulting filter is computationally cheaper and behaviorally identical to the recursive KF once the latter has converged — which for this system occurs within roughly 20 frames at 60 fps.