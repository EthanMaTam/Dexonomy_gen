"""闭环手的连接约束残差、雅可比和被动关节投影，供标注与验收复用。"""

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


def connect_errors(model, data):
    """Each row is body1's anchor minus body2's anchor, in metres."""
    result = []
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT:
            continue  # Scene assembly adds a separate mocap weld for the free hand.
        first, second = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        a = data.xpos[first] + data.xmat[first].reshape(3, 3) @ model.eq_data[i, :3]
        b = data.xpos[second] + data.xmat[second].reshape(3, 3) @ model.eq_data[i, 3:6]
        result.append(a-b)
    return np.asarray(result).reshape(-1, 3)


def connect_jacobian(model, data):
    result = []
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        jacobians = []
        for body, anchor in [(int(model.eq_obj1id[i]), model.eq_data[i, :3]),
                             (int(model.eq_obj2id[i]), model.eq_data[i, 3:6])]:
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ anchor
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, body)
            jacobians.append(jac)
        result.append(jacobians[0]-jacobians[1])
    return np.concatenate(result, axis=0)


def moving_indices(model):
    # Source tip pivots are coincident point joints without visual geometry;
    # their spin doesn't change the anchor position. Keep them at zero.
    return np.array([i for i in range(model.nq) if "tip_pivot" not in model.joint(i).name])


def project_passive(model, qpos, tolerance=2e-5):
    """Hold actual motor targets fixed and solve the transmission coordinates."""
    qpos = np.asarray(qpos, dtype=float).copy()
    active = set(int(i) for i in model.actuator_trnid[:, 0])
    passive = np.array([i for i in moving_indices(model) if i not in active])
    data = mujoco.MjData(model)

    def forward(value):
        data.qpos[:] = qpos
        data.qpos[passive] = value
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    def residual(value):
        forward(value)
        return connect_errors(model, data).ravel()

    def jacobian(value):
        forward(value)
        return connect_jacobian(model, data)[:, passive]

    fitted = least_squares(residual, np.clip(qpos[passive], model.jnt_range[passive, 0]+1e-9,
                                           model.jnt_range[passive, 1]-1e-9), jac=jacobian,
                           bounds=(model.jnt_range[passive, 0], model.jnt_range[passive, 1]),
                           max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    qpos[passive] = fitted.x
    forward(fitted.x)
    error = float(np.linalg.norm(connect_errors(model, data), axis=1).max())
    if error > tolerance:
        raise ValueError(f"Motor targets cannot close the linkage: max anchor error {error:.6g} m")
    return qpos


@lru_cache(maxsize=2)
def _standalone_model(path, mtime):
    return mujoco.MjModel.from_xml_path(path)


def project_free_hand_grasp(xml_path, qpos, limit):
    """Repair excessive loop error without changing the free root or motor angles.

    Called before collision/contact/QP filtering, so projection cannot bypass
    any of the original grasp acceptance conditions.
    """
    path = Path(xml_path).resolve()
    model = _standalone_model(str(path), path.stat().st_mtime_ns)
    qpos = np.asarray(qpos, dtype=float).copy()
    if qpos.shape != (model.nq+7,):
        raise ValueError("Expected a floating hand without movable object coordinates")
    state = mujoco.MjData(model); state.qpos[:] = qpos[7:]
    mujoco.mj_kinematics(model, state)
    before = float(np.linalg.norm(connect_errors(model, state), axis=1).max())
    if before <= limit:
        return qpos, False
    original = qpos.copy()
    full = project_passive(model, qpos[7:], tolerance=min(limit*.5, 2e-5))
    # Full projection can move a nearly acceptable contact more than needed.
    # Find the smallest correction along that feasible direction, with 10%
    # clearance inside the unchanged engineering limit.
    low, high = 0., 1.
    delta = full-qpos[7:]
    for _ in range(30):
        fraction = (low+high)/2
        state.qpos[:] = original[7:]+fraction*delta
        mujoco.mj_kinematics(model, state)
        error = float(np.linalg.norm(connect_errors(model, state), axis=1).max())
        if error <= .9*limit:
            high = fraction
        else:
            low = fraction
    qpos[7:] = original[7:]+high*delta
    active = model.actuator_trnid[:, 0]
    if not np.array_equal(qpos[:7], original[:7]) or not np.array_equal(qpos[7:][active], original[7:][active]):
        raise AssertionError("Projection changed root or motor targets")
    return qpos, True
