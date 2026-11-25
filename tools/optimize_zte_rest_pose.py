"""改进未参与接触手指的收拢姿态，保持接触拓扑和指节局部接触不变。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from ruamel.yaml import YAML

from dexonomy.util.file_util import load_yaml
from tools.closed_loop_hand import project_passive, connect_errors
from tools.prepare_zte_hand import ROOT, DESTINATION
from tools.sharpa_template_editor import _contact_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/zte_rest_tuning/candidates")
    parser.add_argument("--templates", nargs="+", default=["20_Writing_Tripod", "21_Tripod_Variation", "33_Inferior_Pincer"])
    parser.add_argument("--spread-unused", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.yaml")): parser.error("候选目录非空")
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml")); state = mujoco.MjData(model)
    yaml = YAML(); yaml.default_flow_style = None
    records = []
    for name in args.templates:
        for label, source, curl in [
            ("original_tuck_120", DESTINATION/"raw_anno"/(name+".yaml"), 1.2),
            ("selected_tuck_080", ROOT/"output/zte_annotation_tuning/selected_annotations"/(name+".yaml"), .8),
            ("selected_tuck_120", ROOT/"output/zte_annotation_tuning/selected_annotations"/(name+".yaml"), 1.2),
        ]:
            baseline = load_yaml(str(source)); qpos = np.array(baseline["qpos"])
            active = {b.split("_")[0] for b in baseline["contact"] if b != "hand_base_link"}
            state.qpos[:] = qpos; mujoco.mj_forward(model, state)
            local = {}
            for body, values in baseline["contact"].items():
                bid = model.body(body).id; rotation = state.xmat[bid].reshape(3, 3)
                local[body] = [(rotation.T@(np.array(v[:3])-state.xpos[bid]), rotation.T@np.array(v[3:]))
                               for v in _contact_entries(values)]
            for finger in {"index", "middle", "ring", "pinky"}-active:
                spread = {"index": -.2, "middle": 0., "ring": .12, "pinky": .25}[finger] if args.spread_unused else 0.
                for segment, value in [("metacarpal", spread), ("proximal", curl), ("distal", .8)]:
                    qpos[model.joint(finger+"_"+segment+"_joint").id] = value
            qpos = project_passive(model, qpos)
            state.qpos[:] = qpos; mujoco.mj_forward(model, state)
            penetration = max([0.]+[-float(c.dist) for c in state.contact])
            if penetration > 1e-5:
                print(name, label, "reject self-collision", penetration, flush=True); continue
            contacts = {}
            for body, values in local.items():
                bid = model.body(body).id; rotation = state.xmat[bid].reshape(3, 3)
                entries = [np.round(np.r_[rotation@p+state.xpos[bid], rotation@n], 8).tolist() for p, n in values]
                contacts[body] = entries[0] if len(entries) == 1 else entries
            payload = {"qpos": np.round(qpos, 8).tolist(), "contact": contacts}
            if "required_cbody" in baseline: payload["required_cbody"] = baseline["required_cbody"]
            candidate = name+"__"+label+("_spread" if args.spread_unused else ""); path = args.output/(candidate+".yaml")
            with path.open("x") as stream: yaml.dump(payload, stream)
            records.append({"template": name, "candidate": candidate,
                            "baseline_sha256": hashlib.sha256((DESTINATION/"raw_anno"/(name+".yaml")).read_bytes()).hexdigest(),
                            "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "max_connect_error_m": float(np.linalg.norm(connect_errors(model, state), axis=1).max()),
                            "noncontact_finger_curl_rad": curl})
            print(candidate, flush=True)
    (args.output.parent/"candidate_provenance.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__": main()
