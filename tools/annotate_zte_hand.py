"""复用 O30 全分类标注，在 ZTE 碰撞凸包上重定位接触并拟合闭环可行手势。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from ruamel.yaml import YAML
from scipy.optimize import least_squares

from dexonomy.util.file_util import load_yaml
from tools.annotate_o30_taxonomy import local_meshes
from tools.closed_loop_hand import connect_errors, connect_jacobian, moving_indices, project_passive
from tools.prepare_zte_hand import ROOT, DESTINATION
from tools.sharpa_template_editor import _contact_entries

BODY_MAP = {"hand_base_link": "hand_base_link", "thumb_metacarpals_base1": "thumb_metacarpal",
            "thumb_metacarpals_base2": "thumb_proximal", "thumb_proximal": "thumb_middle",
            "thumb_distal": "thumb_distal"}
for finger in ("index", "middle", "ring", "pinky"):
    BODY_MAP[finger+"_metacarpals"] = finger+"_metacarpal"
    for segment in ("proximal", "middle", "distal"):
        BODY_MAP[finger+"_"+segment] = finger+"_"+segment
THUMB_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-evaluations", type=int, default=180)
    parser.add_argument("--templates", nargs="+")
    parser.add_argument("--output", type=Path, default=DESTINATION / "raw_anno")
    parser.add_argument("--legacy-contact-only", action="store_true", help="Reproduce historical O30-to-ZTE contact-only migration")
    args = parser.parse_args()
    if not args.legacy_contact_only:
        from tools.retarget_shadow_annotations import main as retarget
        options=["--hand", "zte_hand", "--output", str(args.output),
                 "--max-evaluations", str(args.max_evaluations), "--skip-existing"]
        if args.templates: options += ["--templates", *args.templates]
        retarget(options)
        return
    source_directory = ROOT / "assets/hand/linker_o30_right"
    source = mujoco.MjModel.from_xml_path(str(source_directory / "right.xml"))
    target = mujoco.MjModel.from_xml_path(str(DESTINATION / "right.xml"))
    sd, td = mujoco.MjData(source), mujoco.MjData(target)
    sm, tm = local_meshes(source, sd), local_meshes(target, td)
    sx0, sr0 = sd.xpos.copy(), sd.xmat.copy().reshape(-1, 3, 3)
    tx0, tr0 = td.xpos.copy(), td.xmat.copy().reshape(-1, 3, 3)
    target.geom_margin[target.geom_contype != 0] = .0005
    indices = moving_indices(target)
    pair_ids = {(a, b): i for i, (a, b) in enumerate((a, b) for a in range(target.nbody) for b in range(a+1, target.nbody))}
    yaml = YAML()
    yaml.default_flow_style = None
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.parent / "taxonomy_metadata.json"
    metadata = {
        "source_hand": "linker_o30_right",
        "source_model_sha256": hashlib.sha256((source_directory/"right.xml").read_bytes()).hexdigest(),
        "target_hand": "zte_hand",
        "target_model_sha256": hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest(),
        "contact_body_map": BODY_MAP,
        "annotation_coordinates": "手基座坐标系，位置单位米，单位外法向；40 个关节角单位弧度",
        "method": "逐指节接触凸包投影、解析雅可比有界姿态拟合、连接约束及自碰撞惩罚、固定电机目标的被动关节投影",
        "closed_loop_annotation_limit_m": 2e-5,
        "program_generated": True, "human_classification_review_complete": False,
    }
    if metadata_path.exists() and list(args.output.glob("*.yaml")):
        old = json.loads(metadata_path.read_text())
        for key in ("source_model_sha256", "target_model_sha256"):
            if old[key] != metadata[key]:
                parser.error("模型已改变，请使用新的标注输出目录，避免混用旧标注与新运动学")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    provenance_path = args.output.parent / "taxonomy_provenance.json"
    previous = {r["template"]: r for r in json.loads(provenance_path.read_text())} if provenance_path.exists() else {}
    reports = []
    paths = sorted((source_directory / "raw_anno").glob("*.yaml"))
    if args.templates:
        paths = [p for p in paths if p.stem in args.templates]
        assert {p.stem for p in paths} == set(args.templates)
    for path in paths:
        destination = args.output / path.name
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            old = previous.get(path.stem, {})
            reports.append(old if old.get("sha256") == digest else {"template": path.stem, "status": "保留已有标注", "sha256": digest})
            continue
        annotation = load_yaml(str(path))
        sd.qpos[:] = annotation["qpos"]
        mujoco.mj_forward(source, sd)
        items = []
        for source_body, raw in annotation["contact"].items():
            name = BODY_MAP[source_body]
            sb, tb = source.body(source_body).id, target.body(name).id
            rotation = THUMB_ROTATION if source_body.startswith("thumb_") else np.eye(3)
            low, high = (sm[source_body].vertices @ rotation.T).min(0), (sm[source_body].vertices @ rotation.T).max(0)
            tlow, thigh = tm[name].bounds
            for entry in _contact_entries(raw):
                point = (np.array(entry[:3])-sd.xpos[sb]) @ sd.xmat[sb].reshape(3, 3)
                normal = np.array(entry[3:]) @ sd.xmat[sb].reshape(3, 3)
                guess = tlow + np.clip((rotation @ point-low)/np.maximum(high-low, 1e-6), 0., 1.)*(thigh-tlow)
                projected, _, faces = tm[name].nearest.on_surface([guess])
                cp, cn = projected[0], tm[name].face_normals[faces[0]]
                expected = rotation @ normal
                if cn @ expected < .3:
                    eligible = np.flatnonzero(tm[name].face_normals @ expected > .6)
                    assert len(eligible), (path, name)
                    face = eligible[np.argmin(np.linalg.norm(tm[name].triangles_center[eligible]-guess, axis=1))]
                    cp, cn = tm[name].triangles_center[face], tm[name].face_normals[face]
                posed_source = sd.xmat[sb].reshape(3, 3) @ point + sd.xpos[sb]
                neutral_source = sr0[sb] @ point + sx0[sb]
                goal = tr0[tb] @ cp + tx0[tb] + .9*(posed_source-neutral_source)
                goal_normal = sd.xmat[sb].reshape(3, 3) @ sr0[sb].T @ tr0[tb] @ cn
                items.append((name, tb, cp, cn, goal, goal_normal))
        initial = np.zeros(target.nq)
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for target_segment, source_suffix in [("metacarpal", "mcp_roll"), ("proximal", "mcp_pitch"), ("distal", "dip")]:
                # Thumb names in O30 are special; its first two coordinates are opposition/flexion.
                source_id = (source.joint(finger+"_"+source_suffix).id if finger != "thumb" else
                             {"metacarpal": 0, "proximal": 1, "distal": 3}[target_segment])
                target_id = target.joint(finger+"_"+target_segment+"_joint").id
                initial[target_id] = sd.qpos[source_id]
        initial = np.clip(initial, target.jnt_range[:, 0]+1e-7, target.jnt_range[:, 1]-1e-7)
        try:
            initial = project_passive(target, initial)
        except ValueError:
            initial[:] = 0.
        collision_values = np.zeros(len(pair_ids))
        collision_jac = np.zeros((len(pair_ids), target.nv))

        def forward(value):
            td.qpos[:] = initial
            td.qpos[indices] = value
            mujoco.mj_forward(target, td)

        def collision_terms():
            collision_values[:] = 0.
            collision_jac[:] = 0.
            for c in td.contact:
                a, b = int(target.geom_bodyid[c.geom1]), int(target.geom_bodyid[c.geom2])
                pair = pair_ids[tuple(sorted((a, b)))]
                depth = .0003-c.dist
                if depth <= collision_values[pair]:
                    continue
                collision_values[pair] = depth
                ja, jb = np.zeros((3, target.nv)), np.zeros((3, target.nv))
                mujoco.mj_jac(target, td, ja, None, c.pos, a)
                mujoco.mj_jac(target, td, jb, None, c.pos, b)
                collision_jac[pair] = -c.frame[:3] @ (jb-ja)

        def residual(value):
            forward(value)
            contact = []
            for _, bid, point, normal, goal, goal_normal in items:
                rotation = td.xmat[bid].reshape(3, 3)
                contact.extend(60*(rotation @ point+td.xpos[bid]-goal))
                contact.extend(.6*(rotation @ normal-goal_normal))
            collision_terms()
            return np.r_[contact, 20000*connect_errors(target, td).ravel(),
                         .03*(value-initial[indices]), 1500*collision_values]

        def jacobian(value):
            forward(value)
            contact = []
            for _, bid, point, normal, _, _ in items:
                rotation = td.xmat[bid].reshape(3, 3)
                ja, jr = np.zeros((3, target.nv)), np.zeros((3, target.nv))
                mujoco.mj_jac(target, td, ja, jr, rotation @ point+td.xpos[bid], bid)
                contact.extend([60*ja, -.6*skew(rotation @ normal) @ jr])
            collision_terms()
            return np.concatenate([np.concatenate(contact)[:, indices],
                                   20000*connect_jacobian(target, td)[:, indices],
                                   .03*np.eye(len(indices)), 1500*collision_jac[:, indices]])

        fitted = least_squares(residual, np.clip(initial[indices], target.jnt_range[indices, 0]+1e-8,
                                                target.jnt_range[indices, 1]-1e-8), jac=jacobian,
                               bounds=(target.jnt_range[indices, 0], target.jnt_range[indices, 1]),
                               max_nfev=args.max_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        forward(fitted.x)
        qpos = project_passive(target, td.qpos)
        td.qpos[:] = qpos
        mujoco.mj_forward(target, td)
        grouped = {}
        for name, bid, point, normal, _, _ in items:
            rotation = td.xmat[bid].reshape(3, 3)
            grouped.setdefault(name, []).append(np.round(np.r_[rotation @ point+td.xpos[bid], rotation @ normal], 8).tolist())
        payload = {"qpos": np.round(qpos, 8).tolist(),
                   "contact": {n: v[0] if len(v) == 1 else v for n, v in grouped.items()}}
        if "required_cbody" in annotation:
            payload["required_cbody"] = [[BODY_MAP[n] for n in group] for group in annotation["required_cbody"]]
        with destination.open("x") as stream:
            yaml.dump(payload, stream)
        report = {"template": path.stem, "status": "闭环约束自动迁移，分类语义待人工审阅",
                  "source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "contact_points": len(items),
                  "fit_cost": float(fitted.cost), "fit_evaluations": fitted.nfev,
                  "max_connect_error_m": float(np.linalg.norm(connect_errors(target, td), axis=1).max()),
                  "max_self_penetration_m": max([0.] + [-float(c.dist) for c in td.contact]),
                  "max_reference_fit_error_m": max(float(np.linalg.norm(td.xmat[b].reshape(3, 3) @ p+td.xpos[b]-g)) for _, b, p, _, g, _ in items)}
        reports.append(report)
        print(path.stem, "contacts", len(items), "loop", f"{report['max_connect_error_m']:.2g}",
              "penetration", f"{report['max_self_penetration_m']:.2g}", flush=True)
        previous[path.stem] = report
        provenance_path.write_text(json.dumps(list(previous.values()), ensure_ascii=False, indent=2)+"\n")
    # Preserve provenance outside a partial selection, and never erase previous records.
    previous.update({r["template"]: r for r in reports})
    provenance_path.write_text(json.dumps(list(previous.values()), ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
