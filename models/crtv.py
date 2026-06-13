import numpy as np

def transition(x, dt, eps=1e-4):
    """
    CTRV state transition function.
    Args:
        x  : state vector [px, py, v, theta, omega] shape (5,)
        dt : time step
    Returns:
        x_next: propagated state shape (5,)
    """
    px, py, v, theta, omega = x

    if abs(omega) > eps:
        x_next = np.array([
            px + (v/omega) * (np.sin(theta + omega*dt) - np.sin(theta)),
            py + (v/omega) * (-np.cos(theta + omega*dt) + np.cos(theta)),
            v,
            theta + omega*dt,
            omega
        ])
    else:
        x_next = np.array([
            px + v * np.cos(theta) * dt,
            py + v * np.sin(theta) * dt,
            v,
            theta,
            omega
        ])

    return x_next


def observation(x):
    """
    CTRV observation function — GPS measures position only.
    Args:
        x: state vector [px, py, v, theta, omega] shape (5,)
    Returns:
        y: observation vector [px, py] shape (2,)
    """
    return x[:2]