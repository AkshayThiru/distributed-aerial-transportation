import numpy as np
import pinocchio as pin


# From Eqs. (10), (11), and (16) in
# T. Lee, M. Leok, and N. McClamroch, “Geometric tracking control of
# a quadrotor UAV on SE(3),” in Proceedings of the IEEE Conference
# on Decision and Control, 2010, pp. 5420–5425.
def so3_pd_tracking_control(
    R: np.ndarray,
    Rd: np.ndarray,
    w: np.ndarray,
    wd: np.ndarray,
    dwd: np.ndarray,
    J: np.ndarray,
    params: tuple[float, float],
) -> np.ndarray:
    assert R.shape == (3, 3) and Rd.shape == (3, 3)
    assert w.shape == (3,) and wd.shape == (3,)
    assert dwd.shape == (3,)
    assert J.shape == (3, 3)
    k_R, k_Omega = params
    assert k_R > 0 and k_Omega > 0

    e_R = 1 / 2 * pin.unSkew(Rd.T @ R - R.T @ Rd)
    e_Omega = w - R.T @ Rd @ wd

    M = (
        -k_R * e_R
        - k_Omega * e_Omega
        + np.cross(w, J @ w)
        - J @ (pin.skew(w) @ R.T @ Rd @ wd - R.T @ Rd @ dwd)
    )
    return M


# From Eqs. (34), (35), and (36) in
# T. Lee, "Geometric Control of Quadrotor UAVs Transporting a
# Cable-Suspended Rigid Body," in IEEE Transactions on
# Control Systems Technology, vol. 26, no. 1, pp. 255-264, Jan. 2018.
def so3_sm_tracking_control(
    R: np.ndarray,
    Rd: np.ndarray,
    w: np.ndarray,
    wd: np.ndarray,
    dwd: np.ndarray,
    J: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    assert R.shape == (3, 3) and Rd.shape == (3, 3)
    assert w.shape == (3,) and wd.shape == (3,)
    assert dwd.shape == (3,)
    assert J.shape == (3, 3)
    r, k_R, l_R, k_s, l_s = params
    assert r > 0 and r < 1
    assert k_R > 0 and l_R > 0
    assert k_s > 0 and l_s > 0

    e_R = 1 / 2 * pin.unSkew(Rd.T @ R - R.T @ Rd)
    e_Omega = w - R.T @ Rd @ wd
    E = 1 / 2 * (np.trace(R.T @ Rd) * np.eye(3) - R.T @ Rd)
    S = lambda r, y: np.power(np.abs(y), r) * np.sign(y)
    s = e_Omega + k_R * e_R + l_R * S(r, e_R)

    eps = 1e-6
    T = lambda r, y: np.diag(np.power(np.abs(y) + eps, r - 1))
    M = (
        -k_s * s
        - l_s * S(r, s)
        + np.cross(w, J @ w)
        - (k_R * J + l_s * r * (J @ T(e_R, r))) @ E @ e_Omega
        - J @ (pin.skew(w) @ R.T @ Rd @ wd - R.T @ Rd @ dwd)
    )
    return M
