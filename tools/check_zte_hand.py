"""核对 ZTE 转换前后的运动学、连接雅可比和自碰撞开关。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.closed_loop_hand import connect_errors, connect_jacobian
from tools.prepare_zte_hand import ROOT, SOURCE, DESTINATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT/"output/zte_model_checks.json")
    args = parser.parse_args()
    metadata = json.loads((DESTINATION/"provenance.json").read_text())
    source = mujoco.MjModel.from_xml_path(str(SOURCE/"xml/right_hand_v1.0.xml"))
    target = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml"))
    assert (target.nq, target.nu, target.neq) == (40, 15, 10)
    assert np.all(target.eq_active0), "A transmission constraint is disabled"
    assert target.nexclude == 10 and source.nexclude == 253
    assert np.all(target.actuator_forcelimited) and np.allclose(target.actuator_forcerange, [-1, 1])
    sd, td = mujoco.MjData(source), mujoco.MjData(target)
    random = np.random.default_rng(20261121)
    source_root = source.body("arm_Y_Link").id
    shift = np.array(metadata["root_shift_in_source_arm_Y_frame_m"])
    max_position_error, max_rotation_error = 0., 0.
    for _ in range(8):
        td.qpos[:] = random.uniform(target.jnt_range[:, 0], target.jnt_range[:, 1])
        sd.qpos[:] = 0.
        for old, new in metadata["joint_name_map"].items():
            sd.qpos[source.joint(old).qposadr] = td.qpos[target.joint(new).qposadr]
        mujoco.mj_forward(source, sd)
        mujoco.mj_forward(target, td)
        rotation = sd.xmat[source_root].reshape(3, 3)
        origin = sd.xpos[source_root] + rotation @ shift
        for old, new in metadata["body_name_map"].items():
            if new == "hand_base_link":
                continue
            a, b = source.body(old).id, target.body(new).id
            max_position_error = max(max_position_error, float(np.max(np.abs((sd.xpos[a]-origin) @ rotation-td.xpos[b]))))
            max_rotation_error = max(max_rotation_error, float(np.max(np.abs(rotation.T @ sd.xmat[a].reshape(3, 3)-td.xmat[b].reshape(3, 3)))))
    assert max_position_error < 1e-9 and max_rotation_error < 1e-9
    # Cross-finger pairs remain enabled. Internal transmission rods are visual only.
    excluded = {tuple(sorted((target.body(int(v >> 16)).name, target.body(int(v & 65535)).name)))
                for v in target.exclude_signature}
    assert all("hand_base_link" in pair for pair in excluded)
    target_indices = [target.joint("index_distal_joint").id, target.joint("thumb_proximal_joint").id]
    td.qpos[:] = 0.
    td.qpos[target_indices] = [.4, .3]
    mujoco.mj_forward(target, td)
    jac = connect_jacobian(target, td)
    max_jacobian_error = 0.
    epsilon = 1e-7
    for i in range(target.nq):
        base = td.qpos[i]
        td.qpos[i] = base+epsilon
        mujoco.mj_forward(target, td)
        plus = connect_errors(target, td).ravel()
        td.qpos[i] = base-epsilon
        mujoco.mj_forward(target, td)
        minus = connect_errors(target, td).ravel()
        td.qpos[i] = base
        max_jacobian_error = max(max_jacobian_error, float(np.max(np.abs((plus-minus)/(2*epsilon)-jac[:, i]))))
    assert max_jacobian_error < 1e-7
    td.qpos[:] = 0.
    mujoco.mj_forward(target, td)
    assert all(c.dist >= -1e-7 for c in td.contact)
    report = {"source_archive_sha256": metadata["archive_sha256"],
              "model_sha256": hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest(),
              "nq": target.nq, "actuators": target.nu, "connect_equalities": target.neq,
              "source_kinematics_verified_poses": 8,
              "max_position_error_m": max_position_error, "max_rotation_matrix_error": max_rotation_error,
              "connect_jacobian_max_error": max_jacobian_error,
              "rest_pose_self_collision_free": True, "cross_finger_self_collision_enabled": True,
              "structural_exclusions": target.nexclude}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
