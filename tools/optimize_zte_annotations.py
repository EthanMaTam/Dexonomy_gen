"""为 ZTE 未通过分类生成指腹重定位和闭环虚拟球拟合候选，不改接触拓扑。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from ruamel.yaml import YAML

from dexonomy.util.file_util import load_yaml
from tools.annotate_o30_taxonomy import local_meshes
from tools.annotate_zte_hand import skew
from tools.closed_loop_hand import connect_errors, connect_jacobian, project_passive, moving_indices
from tools.prepare_zte_hand import ROOT, DESTINATION
from tools.sharpa_template_editor import _contact_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/zte_annotation_tuning/candidates")
    parser.add_argument("--max-evaluations", type=int, default=250)
    parser.add_argument("--templates", nargs="+")
    parser.add_argument("--shape", choices=["sphere", "cylinder"], default="sphere")
    parser.add_argument("--annotation-dir", type=Path, default=DESTINATION/"raw_anno")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.yaml")):
        parser.error("候选目录非空，请使用新目录")
    targets = json.loads((DESTINATION/"validation_summary.json").read_text())["needs_improvement"]
    if args.templates:
        assert set(args.templates) <= set(targets)
        targets = args.templates
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml"))
    state = mujoco.MjData(model)
    meshes = local_meshes(model, state)
    model.geom_margin[model.geom_contype != 0] = .0005
    pairs = {(a, b): i for i, (a, b) in enumerate((a, b) for a in range(model.nbody) for b in range(a+1, model.nbody))}
    yaml = YAML(); yaml.default_flow_style = None
    records = []
    for name in targets:
        source = args.annotation_dir/(name+".yaml")
        baseline = load_yaml(str(source)); base_q = np.array(baseline["qpos"])
        pads = []
        for body, raw in baseline["contact"].items():
            mesh = meshes[body]
            entries = _contact_entries(raw)
            low, high = mesh.bounds
            for i, entry in enumerate(entries):
                center = (low+high)/2
                center[2] = low[2]+(.65 if body.endswith("distal") else .55)*(high[2]-low[2])
                center[2] += (i-(len(entries)-1)/2)*.008
                axis = np.array([0., 1., 0.]) if name == "16_Lateral" and body == "index_middle" else np.array([1., 0., 0.])
                # Palm contacts use the broad middle of the palm, not a finger bearing.
                if body == "hand_base_link":
                    center[1] = (i-(len(entries)-1)/2)*.016
                    center[2] = .045
                hits, _, faces = mesh.ray.intersects_location([center+.2*axis], [-axis], multiple_hits=False)
                assert len(hits) == 1, (name, body)
                cp, cn = hits[0], mesh.face_normals[faces[0]]
                pads.append((body, model.body(body).id, cp, cn))
        indices = np.array([i for i in moving_indices(model) if model.joint(i).name.split("_")[0] in
                            {b.split("_")[0] for b in baseline["contact"] if b != "hand_base_link"}])

        def payload(qpos):
            state.qpos[:] = qpos; mujoco.mj_forward(model, state)
            grouped = {}
            for body, bid, point, normal in pads:
                rotation = state.xmat[bid].reshape(3, 3)
                grouped.setdefault(body, []).append(np.round(np.r_[rotation@point+state.xpos[bid], rotation@normal], 8).tolist())
            result = {"qpos": np.round(qpos, 8).tolist(),
                      "contact": {body: values[0] if len(values) == 1 else values for body, values in grouped.items()}}
            if "required_cbody" in baseline:
                result["required_cbody"] = baseline["required_cbody"]
            return result

        def store(suffix, qpos, **details):
            result = payload(qpos)
            assert list(result["contact"]) == list(baseline["contact"])
            assert all(len(_contact_entries(result["contact"][b])) == len(_contact_entries(baseline["contact"][b])) for b in result["contact"])
            candidate = name+"__"+suffix
            path = args.output/(candidate+".yaml")
            with path.open("x") as stream: yaml.dump(result, stream)
            record = {"candidate": candidate, "template": name,
                      "baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "contact_topology_preserved": True,
                      "max_connect_error_m": float(np.linalg.norm(connect_errors(model, state), axis=1).max()),
                      "max_self_penetration_m": max([0.]+[-float(c.dist) for c in state.contact]), **details}
            records.append(record)
            print(candidate, "loop", record["max_connect_error_m"], "penetration", record["max_self_penetration_m"], flush=True)
            (args.output.parent/"candidate_provenance.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")

        if args.shape == "sphere":
            store("pad_center", base_q, method="保留姿态，将接触重定位到指腹；Lateral 保留食指侧面")
        for radius in (.025, .045, .065):
            state.qpos[:] = base_q; mujoco.mj_forward(model, state)
            axis = np.array([0., 1., 0.])
            if args.shape == "cylinder" and name == "33_Inferior_Pincer":
                points = np.array([state.xmat[b].reshape(3, 3)@p+state.xpos[b] for _, b, p, _ in pads])
                axis = np.linalg.svd(points-points.mean(0), full_matrices=False)[2][0]
            projection = np.eye(3)-np.outer(axis, axis) if args.shape == "cylinder" else np.eye(3)
            center0 = np.mean([state.xmat[b].reshape(3, 3)@(p+radius*n)+state.xpos[b] for _, b, p, n in pads], axis=0)
            center0 = np.clip(center0, [.015, -.04, .04], [.14, .07, .17])

            def forward(value):
                state.qpos[:] = base_q; state.qpos[indices] = value[:-3]
                mujoco.mj_forward(model, state)

            def collisions():
                values = np.zeros(len(pairs)); jac = np.zeros((len(pairs), model.nv))
                for c in state.contact:
                    a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                    index = pairs[tuple(sorted((a, b)))]; depth = .0003-c.dist
                    if depth <= values[index]: continue
                    values[index] = depth
                    ja, jb = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jac(model, state, ja, None, c.pos, a)
                    mujoco.mj_jac(model, state, jb, None, c.pos, b)
                    jac[index] = -c.frame[:3]@(jb-ja)
                return values, jac

            def residual(value):
                forward(value)
                surface = np.ravel([80*projection@(state.xmat[b].reshape(3, 3)@(p+radius*n)+state.xpos[b]-value[-3:]) for _, b, p, n in pads])
                normal = ([.6*axis@(state.xmat[b].reshape(3, 3)@n) for _, b, _, n in pads] if args.shape == "cylinder" else [])
                return np.r_[surface, normal, 20000*connect_errors(model, state).ravel(),
                             .08*(value[:-3]-base_q[indices]), .1*(value[-3:]-center0), 1500*collisions()[0]]

            def jacobian(value):
                forward(value)
                contacts = []
                for _, bid, p, n in pads:
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    point = state.xmat[bid].reshape(3, 3)@(p+radius*n)+state.xpos[bid]
                    mujoco.mj_jac(model, state, jp, jr, point, bid)
                    contacts.append(np.c_[80*projection@jp[:, indices], -80*projection])
                normals = []
                if args.shape == "cylinder":
                    for _, bid, p, n in pads:
                        jr = np.zeros((3, model.nv)); jp = np.zeros_like(jr)
                        mujoco.mj_jac(model, state, jp, jr, state.xpos[bid], bid)
                        normal = state.xmat[bid].reshape(3, 3)@n
                        normals.append(np.r_[.6*axis@(-skew(normal)@jr[:, indices]), np.zeros(3)][None])
                return np.concatenate([*contacts, *normals, np.c_[20000*connect_jacobian(model, state)[:, indices], np.zeros((30, 3))],
                                       np.c_[.08*np.eye(len(indices)), np.zeros((len(indices), 3))],
                                       np.c_[np.zeros((3, len(indices))), .1*np.eye(3)],
                                       np.c_[1500*collisions()[1][:, indices], np.zeros((len(pairs), 3))]])
            low = np.r_[model.jnt_range[indices, 0], [.015, -.04, .04]]
            high = np.r_[model.jnt_range[indices, 1], [.14, .07, .17]]
            fit = least_squares(residual, np.clip(np.r_[base_q[indices], center0], low+1e-8, high-1e-8),
                                jac=jacobian, bounds=(low, high), max_nfev=args.max_evaluations,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            forward(fit.x)
            try:
                qpos = project_passive(model, state.qpos)
            except ValueError as error:
                print(name, radius, "skip", str(error), flush=True); continue
            store(f"{args.shape}_{int(radius*1000):03}", qpos, radius_m=radius, fit_cost=float(fit.cost),
                  method=f"保持接触拓扑与闭环约束，拟合虚拟 {args.shape}，带自碰撞惩罚")


if __name__ == "__main__": main()
