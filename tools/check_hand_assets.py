"""Check hand actuator coverage, joint limits and annotated collision surfaces."""

import argparse
from pathlib import Path
import json

import mujoco
import numpy as np

from dexonomy.sim import HandCfg, MuJoCo_VisEnv
from dexonomy.util.file_util import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True)
    parser.add_argument("--source-xml", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    directory = ROOT / "assets/hand" / args.hand
    cfg = load_yaml(str(ROOT / "dexonomy/config/hand" / f"{args.hand}.yaml"))
    xml = ROOT / cfg["xml_path"].replace("${asset_root}", "assets")
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nu > 0, "Model has no actuators"
    # Current supported templates describe scalar hinge/slide joints.
    assert all(t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
               for t in model.jnt_type)
    assert set(model.actuator_trnid[:, 0]) == set(range(model.njnt)), "Unactuated joints"
    env = MuJoCo_VisEnv(hand_cfg=HandCfg(str(xml)), vis_mode="collision")
    report = {"hand": args.hand, "nq": model.nq, "actuators": model.nu, "templates": []}
    for path in sorted((directory / "raw_anno").glob("*.yaml")):
        annotation = load_yaml(str(path))
        qpos = np.asarray(annotation["qpos"])
        assert qpos.shape == (model.nq,) and np.isfinite(qpos).all(), path
        assert (qpos >= model.jnt_range[:, 0]-1e-7).all(), path
        assert (qpos <= model.jnt_range[:, 1]+1e-7).all(), path
        rotations, positions = env.forward_kinematics(qpos)
        maximum_distance = 0.
        contacts = 0
        for body, values in annotation["contact"].items():
            values = values if isinstance(values[0], list) else [values]
            bid = env.get_body_id(body)
            mesh = env.get_body_mesh(body)
            for value in values:
                value = np.asarray(value)
                assert value.shape == (6,) and np.isfinite(value).all(), path
                assert abs(np.linalg.norm(value[3:])-1) < 1e-5, path
                point = (value[:3]-positions[bid]) @ rotations[bid]
                _, distance, faces = mesh.nearest.on_surface([point])
                assert distance[0] < 0.002, (path, body, distance)
                normal = value[3:] @ rotations[bid]
                assert normal @ mesh.face_normals[faces[0]] > 0.5, (path, body, "reversed normal")
                maximum_distance = max(maximum_distance, float(distance[0]))
                contacts += 1
        assert contacts >= 2, path
        report["templates"].append({"name": path.stem, "contacts": contacts,
                                    "max_surface_distance_m": maximum_distance})
    assert report["templates"], "No annotations"
    # Compare source and converted kinematics at nonzero poses, not just the rest pose.
    if args.source_xml:
        source = mujoco.MjModel.from_xml_path(str(args.source_xml))
        assert source.nq == model.nq
        a, b = mujoco.MjData(model), mujoco.MjData(source)
        random = np.random.default_rng(12)
        for _ in range(8):
            a.qpos[:] = b.qpos[:] = random.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
            mujoco.mj_forward(model, a)
            mujoco.mj_forward(source, b)
            for bid in range(1, model.nbody):
                sid = source.body(model.body(bid).name).id
                assert np.allclose(a.xpos[bid], b.xpos[sid], atol=1e-10)
                assert np.allclose(a.xmat[bid], b.xmat[sid], atol=1e-10)
        report["source_kinematics_verified_poses"] = 8
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
