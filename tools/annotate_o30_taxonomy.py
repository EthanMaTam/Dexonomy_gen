"""迁移 Shadow 的 33 类抓取标注到 O30 右手，保留已有模板。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from ruamel.yaml import YAML
from scipy.optimize import least_squares

from dexonomy.util.file_util import load_yaml
from tools.sharpa_template_editor import ROOT, _contact_entries, _geom_mesh


BODY_MAP = {"rh_palm": "hand_base_link", "rh_lfmetacarpal": "pinky_metacarpals",
            "rh_thmiddle": "thumb_proximal", "rh_thdistal": "thumb_distal"}
for short, finger in [("ff", "index"), ("mf", "middle"), ("rf", "ring"), ("lf", "pinky")]:
    for old, new in [("proximal", "proximal"), ("middle", "middle"), ("distal", "distal")]:
        BODY_MAP[f"rh_{short}{old}"] = f"{finger}_{new}"

PALM_ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
THUMB_DISTAL_ROTATION = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
THUMB_PROXIMAL_ROTATION = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])


def local_meshes(model, data):
    mujoco.mj_forward(model, data)
    meshes = {}
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0:
            continue
        bid = int(model.geom_bodyid[gid])
        name = model.body(bid).name
        # MuJoCo tests collisions against each mesh's convex hull. Contacts on
        # recessed CAD surfaces would sit inside that hull and are invalid here.
        mesh = _geom_mesh(model, gid).convex_hull
        mesh.vertices = (mesh.vertices @ data.geom_xmat[gid].reshape(3, 3).T
                         + data.geom_xpos[gid] - data.xpos[bid]) @ data.xmat[bid].reshape(3, 3)
        # Bodies in the source can have multiple collision pieces.
        if name in meshes:
            import trimesh
            meshes[name] = trimesh.util.concatenate([meshes[name], mesh])
        else:
            meshes[name] = mesh
    return meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evaluations", type=int, default=200)
    args = parser.parse_args()
    directory = ROOT / "assets/hand/linker_o30_right"
    source_directory = ROOT / "assets/hand/shadow"
    source = mujoco.MjModel.from_xml_path(str(source_directory / "right.xml"))
    target = mujoco.MjModel.from_xml_path(str(directory / "right.xml"))
    sd, td = mujoco.MjData(source), mujoco.MjData(target)
    sm, tm = local_meshes(source, sd), local_meshes(target, td)
    sx0, sr0 = sd.xpos.copy(), sd.xmat.copy().reshape(-1, 3, 3)
    tx0, tr0 = td.xpos.copy(), td.xmat.copy().reshape(-1, 3, 3)
    keypoints = load_yaml(str(source_directory / "keypoint.yaml"))
    for name, value in keypoints.items():
        if isinstance(value, str):
            keypoints[name] = keypoints[value]
    pairs = {(a, b): i for i, (a, b) in enumerate(
        ( (a, b) for a in range(target.nbody) for b in range(a+1, target.nbody) ))}
    # During IK, generate near-contact pairs too, to maintain a small clearance.
    target.geom_margin[target.geom_contype != 0] = 0.001
    output = directory / "raw_anno"
    output.mkdir(exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = None
    reports = []
    provenance = directory / "taxonomy_provenance.json"
    previous = {row["template"]: row for row in json.loads(provenance.read_text())} if provenance.exists() else {}
    paths = sorted([p for p in (source_directory / "raw_anno").glob("*.yaml")
                    if p.stem.split("_")[0].isdigit()], key=lambda p: int(p.stem.split("_")[0]))
    assert len(paths) == 33
    for path in paths:
        destination = output / path.name
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            old = previous.get(path.stem, {})
            reports.append(old if old.get("sha256") == digest else
                           {"template": path.stem, "status": "保留已有标注", "sha256": digest})
            print(path.stem, "保留已有标注", flush=True)
            continue
        annotation = load_yaml(str(path))
        sd.qpos[:] = annotation["qpos"]
        mujoco.mj_forward(source, sd)
        items = []
        for source_body, raw in annotation["contact"].items():
            target_body = BODY_MAP[source_body]
            sbid, tbid = source.body(source_body).id, target.body(target_body).id
            rotation = (THUMB_DISTAL_ROTATION if source_body == "rh_thdistal" else
                        THUMB_PROXIMAL_ROTATION if source_body == "rh_thmiddle" else PALM_ROTATION)
            source_bounds = np.array([sm[source_body].vertices @ rotation.T]).reshape(-1, 3)
            low, high = source_bounds.min(0), source_bounds.max(0)
            tlow, thigh = tm[target_body].bounds
            for entry in _contact_entries(raw):
                if isinstance(entry, list):
                    cp = (np.asarray(entry[:3])-sd.xpos[sbid]) @ sd.xmat[sbid].reshape(3, 3)
                    cn = np.asarray(entry[3:]) @ sd.xmat[sbid].reshape(3, 3)
                else:
                    cp, cn = np.array(keypoints[source_body][int(entry)]).reshape(2, 3)
                fraction = np.clip((rotation @ cp-low)/np.maximum(high-low, 1e-6), 0., 1.)
                guess = tlow + fraction * (thigh-tlow)
                points, _, faces = tm[target_body].nearest.on_surface([guess])
                point = points[0]
                normal = tm[target_body].face_normals[faces[0]]
                # Resolve ambiguous edge projections using the expected anatomical normal.
                expected_normal = rotation @ cn
                expected_normal /= np.linalg.norm(expected_normal)
                if normal @ expected_normal < 0.3:
                    mesh = tm[target_body]
                    eligible = np.flatnonzero(mesh.face_normals @ expected_normal > 0.6)
                    assert len(eligible), (path, target_body)
                    index = eligible[np.argmin(np.linalg.norm(mesh.triangles_center[eligible]-guess, axis=1))]
                    point, normal = mesh.triangles_center[index], mesh.face_normals[index]
                posed_source = sd.xmat[sbid].reshape(3, 3) @ cp + sd.xpos[sbid]
                neutral_source = sr0[sbid] @ cp + sx0[sbid]
                neutral_target = tr0[tbid] @ point + tx0[tbid]
                desired_point = neutral_target + 1.15 * PALM_ROTATION @ (posed_source-neutral_source)
                relative_rotation = sd.xmat[sbid].reshape(3, 3) @ sr0[sbid].T
                desired_normal = PALM_ROTATION @ relative_rotation @ PALM_ROTATION.T @ tr0[tbid] @ normal
                items.append((target_body, tbid, point, normal, desired_point, desired_normal))
        initial = np.zeros(target.nq)
        for short, finger in [("FF", "index"), ("MF", "middle"), ("RF", "ring"), ("LF", "pinky")]:
            for old, new in [(4, "mcp_roll"), (3, "mcp_pitch"), (2, "pip"), (1, "dip")]:
                si, ti = source.joint(f"rh_{short}J{old}").id, target.joint(f"{finger}_{new}").id
                initial[ti] = sd.qpos[si]
        initial[:4] = [0.3 + 0.25*sd.qpos[source.joint("rh_THJ5").id],
                       sd.qpos[source.joint("rh_THJ4").id]*1.4,
                       sd.qpos[source.joint("rh_THJ2").id]+0.5,
                       sd.qpos[source.joint("rh_THJ1").id]]
        low = target.jnt_range[:, 0]+1e-5
        high = target.jnt_range[:, 1]-1e-5
        initial = np.clip(initial, low, high)

        def residual(q):
            td.qpos[:] = q
            mujoco.mj_forward(target, td)
            positions, normals = [], []
            for _, bid, point, normal, desired_point, desired_normal in items:
                r = td.xmat[bid].reshape(3, 3)
                positions.extend(60 * (r @ point + td.xpos[bid]-desired_point))
                normals.extend(0.8 * (r @ normal-desired_normal))
            collision = np.zeros(len(pairs))
            for contact in td.contact:
                a, b = sorted((int(target.geom_bodyid[contact.geom1]), int(target.geom_bodyid[contact.geom2])))
                if target.body_parentid[b] == a or target.body_parentid[a] == b:
                    continue
                collision[pairs[a,b]] = max(collision[pairs[a,b]], 0.0003-contact.dist)
            return np.r_[positions, normals, 0.12*(q-initial), 1500*collision]

        fitted = least_squares(residual, initial, bounds=(low, high),
                               max_nfev=args.max_evaluations, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        residual(fitted.x)
        grouped = {}
        distances = []
        for body, bid, point, normal, desired_point, _ in items:
            r = td.xmat[bid].reshape(3, 3)
            world_point = r @ point + td.xpos[bid]
            distances.append(float(np.linalg.norm(world_point-desired_point)))
            grouped.setdefault(body, []).append(np.round(np.r_[world_point, r @ normal], 7).tolist())
        payload = {"qpos": np.round(fitted.x, 7).tolist(),
                   "contact": {name: values[0] if len(values)==1 else values for name, values in grouped.items()}}
        if "required_cbody" in annotation:
            payload["required_cbody"] = [[BODY_MAP[name] for name in group]
                                          for group in annotation["required_cbody"]]
        with destination.open("x") as stream:
            yaml.dump(payload, stream)
        report = {"template": path.stem, "status": "自动迁移",
                  "validation": "逐类结果见 taxonomy_validation.csv",
                  "source": str(path.relative_to(ROOT)),
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "method": "解剖连杆映射、网格表面投影、有界姿态拟合与自碰撞惩罚",
                  "contact_points": len(items), "max_reference_fit_error_m": max(distances),
                  "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "note": "O30 与 Shadow 自由度和几何不同，分类语义仍需人工审阅；小指掌骨映射到 O30 小指基座。"}
        reports.append(report)
        print(path.stem, len(items), "接触点", flush=True)
    provenance.write_text(json.dumps(reports, ensure_ascii=False, indent=2)+"\n")
    print("已覆盖 33 类编号模板；已有文件保持不变。", flush=True)


if __name__ == "__main__":
    main()
