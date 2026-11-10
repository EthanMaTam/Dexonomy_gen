"""Create geometry-based O30 seed annotations, contact keypoints and skeletons.

Seeds fit fingertip pads to virtual spheres with bounded inverse kinematics.
These are automatically generated seeds, not human annotations or grasp labels.
"""

from pathlib import Path
import argparse
import json

import mujoco
import numpy as np
from scipy.optimize import least_squares
from ruamel.yaml import YAML

from tools.sharpa_template_editor import ROOT, _geom_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing seed annotations")
    args = parser.parse_args()
    destination = ROOT / "assets/hand/linker_o30_right"
    if list((destination / "raw_anno").glob("*.yaml")) and not args.overwrite:
        parser.error("Annotations already exist; use --overwrite only to regenerate the seeds")
    model = mujoco.MjModel.from_xml_path(str(destination / "right.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    yaml = YAML()
    yaml.default_flow_style = None

    def save(name, payload):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            yaml.dump(payload, stream)

    meshes = {}
    skeleton = {}
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0:
            continue
        bid = model.geom_bodyid[gid]
        name = model.body(bid).name
        mesh = _geom_mesh(model, gid)
        # MuJoCo's mesh vertices are in the compiled mesh frame, not link frame.
        mesh.vertices = (
            mesh.vertices @ data.geom_xmat[gid].reshape(3, 3).T
            + data.geom_xpos[gid] - data.xpos[bid]
        ) @ data.xmat[bid].reshape(3, 3)
        meshes[name] = mesh
        low, high = mesh.bounds
        center = (low + high) / 2
        axis = int(np.argmax(high - low))
        start, end = center.copy(), center.copy()
        start[axis] = low[axis] + 0.15 * (high[axis] - low[axis])
        end[axis] = high[axis] - 0.15 * (high[axis] - low[axis])
        skeleton[name] = [np.round(np.r_[start, end], 7).tolist()]
    save("skeleton.yaml", skeleton)
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    save("body_group.yaml", {"body_group": [
        [name for name in meshes if name.startswith(finger + "_")]
        for finger in fingers
    ] + [["hand_base_link"]]})
    pads = {}
    for finger in fingers:
        mesh = meshes[finger + "_distal"]
        axis = np.array([0., 0., 1.]) if finger == "thumb" else np.array([1., 0., 0.])
        center = np.array([0., 0.015, 0.]) if finger == "thumb" else np.array([0., 0., 0.015])
        locations, _, triangles = mesh.ray.intersects_location(
            [center + 0.05 * axis], [-axis], multiple_hits=False
        )
        assert len(locations) == 1, finger
        point, normal = locations[0], mesh.face_normals[triangles[0]]
        assert normal @ axis > 0.5, (finger, normal)
        pads[finger] = (point, normal)
    save("keypoint.yaml", {finger + "_distal": [np.round(np.concatenate(pads[finger]), 7).tolist()]
                           for finger in fingers})
    reports = []
    specs = [
        ("24_Tip_Pinch", ["thumb", "index"], 0.018),
        ("14_Tripod", ["thumb", "index", "middle"], 0.025),
        ("fingertip_large", ["thumb", "index", "middle"], 0.045),
    ]
    pair_indices = {(a, b): i for i, (a, b) in enumerate(
        ( (a, b) for a in range(model.nbody) for b in range(a+1, model.nbody) ))}
    for name, active, radius in specs:
        indices = [i for i in range(model.njnt)
                   if model.joint(i).name.split("_")[0] in active]
        base = np.zeros(model.nq)
        initial = np.tile([0., 0.5, 0.65, 0.35], 5)
        initial[:4] = [0.4, 1.0, 0.6, 0.5]
        initial = np.clip(initial, model.jnt_range[:, 0]+1e-6, model.jnt_range[:, 1]-1e-6)
        center0 = np.array([0.065, 0.015, 0.13])

        def residual(value):
            data.qpos[:] = base
            data.qpos[indices] = value[:-3]
            mujoco.mj_forward(model, data)
            centers = []
            for finger in active:
                bid = model.body(finger + "_distal").id
                rotation = data.xmat[bid].reshape(3, 3)
                point, normal = pads[finger]
                centers.append(rotation @ (point + radius * normal) + data.xpos[bid])
            collision = np.zeros(len(pair_indices))
            for c in data.contact:
                a, b = sorted((int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])))
                if model.body_parentid[b] == a or model.body_parentid[a] == b:
                    continue
                collision[pair_indices[a, b]] = max(collision[pair_indices[a, b]], -c.dist)
            return np.r_[100 * (np.array(centers) - value[-3:]).ravel(),
                         0.015 * (value[:-3] - initial[indices]),
                         0.2 * (value[-3:] - center0), 1000 * collision]

        result = least_squares(
            residual, np.r_[initial[indices], center0],
            bounds=(np.r_[model.jnt_range[indices, 0], [0.03, -0.02, 0.08]],
                    np.r_[model.jnt_range[indices, 1], [0.13, 0.06, 0.18]]),
            max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        residual(result.x)
        contact = {}
        for finger in active:
            bid = model.body(finger + "_distal").id
            rotation = data.xmat[bid].reshape(3, 3)
            point, normal = pads[finger]
            contact[finger + "_distal"] = np.round(
                np.r_[rotation @ point + data.xpos[bid], rotation @ normal], 7
            ).tolist()
        save(f"raw_anno/{name}.yaml", {"qpos": np.round(data.qpos, 7).tolist(), "contact": contact})
        reports.append({"template": name, "method": "bounded IK against a virtual sphere",
                        "radius_m": radius, "max_pad_fit_error_m": float(
                            np.linalg.norm(result.fun[:3*len(active)].reshape(-1, 3), axis=1).max()/100)})
    (destination / "template_provenance.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
