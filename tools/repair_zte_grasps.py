"""投影修复 ZTE 超限抓取，只保留重新通过原碰撞、接触、QP 和外力评估的结果。"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from omegaconf import OmegaConf

from dexonomy.op.gen_grasp import GraspFilter
from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
from dexonomy.util.file_util import load_scene_cfg
from tools.closed_loop_hand import connect_errors
from dexonomy.util.closed_loop_util import project_free_hand_grasp
from tools.prepare_zte_hand import ROOT, DESTINATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"output/zte_hand_validation")
    parser.add_argument("--output", type=Path, default=ROOT/"output/zte_grasp_repair")
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()): parser.error("修复输出已存在，请使用新目录")
    args.output.mkdir(parents=True, exist_ok=True)
    source_job = json.loads((args.run/"job.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml"))
    assert hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest() == source_job["model_sha256"]
    filters = OmegaConf.load(ROOT/"dexonomy/config/op/grasp.yaml")
    records = []
    for row in csv.DictReader((args.run/"closed_loop_samples.csv").open()):
        if row["closed_loop_accepted"] == "True": continue
        source = args.run/row["path"]
        data = copy.deepcopy(np.load(source, allow_pickle=True).item())
        old = data["grasp_qpos"][0].copy()
        new, _ = project_free_hand_grasp(DESTINATION/"right.xml", old, .0002)
        active = model.actuator_trnid[:, 0]
        assert np.array_equal(new[7:][active], old[7:][active])
        scene = load_scene_cfg(data["scene_path"])
        env = MuJoCo_OptEnv(HandCfg(str(DESTINATION/"right.xml"), freejoint=True), scene,
                            MuJoCo_OptCfg(obj_margin=filters.grasp.cdist))
        env.reset_qpos(new)
        valid, contacts = GraspFilter(filters.grasp.filter, env, str(source)).forward(data)
        record = {"source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "template": row["template"], "old_connect_error_m": float(row["max_connect_error_m"]),
                  "grasp_motor_targets_unchanged": True, "original_grasp_filters_passed": bool(valid),
                  "max_coordinate_change_rad": float(np.max(abs(new-old)))}
        if not valid:
            records.append(record); print(record, flush=True); continue
        data["grasp_qpos"] = new[None].astype(np.float32)
        data["ho_c"] = contacts
        data["hand_cbody"] = contacts["bn1"]
        data["hand_cpn_w"] = np.c_[contacts["pos"], contacts["normal"]]
        data["squeeze_qpos"] = env.get_squeeze_qpos(new, contacts["bn1"], contacts["pos"], contacts["wrench"])[None].astype(np.float32)
        env.reset_qpos(data["pregrasp_qpos"][0]); env.set_obj_margin(filters.pregrasp.cdist)
        record["original_pregrasp_filters_passed"] = GraspFilter(filters.pregrasp.filter, env, str(source)).forward(None)[0]
        if not record["original_pregrasp_filters_passed"]:
            records.append(record); print(record, flush=True); continue
        check = mujoco.MjData(model); check.qpos[:] = data["grasp_qpos"][0, 7:]
        mujoco.mj_kinematics(model, check)
        record["new_connect_error_m"] = float(np.linalg.norm(connect_errors(model, check), axis=1).max())
        assert record["new_connect_error_m"] < .0002
        relative = Path(row["path"]).relative_to("shards/0000/succ_grasp")
        destination = args.output/"shards/0000/grasp_data"/relative
        destination.parent.mkdir(parents=True, exist_ok=True); np.save(destination, data)
        record["repaired_path"] = str(destination.relative_to(ROOT))
        records.append(record); print(record, flush=True)
    (args.output/"job.json").write_text(json.dumps({"model_sha256": source_job["model_sha256"],
                                                   "parent_run": str(args.run.relative_to(ROOT)), "repair_only": True}, indent=2)+"\n")
    subprocess.run([sys.executable, "-m", "dexonomy.main", "op=eval", "hand=zte_hand",
                    f"save_dir={args.output}/shards/0000", "n_worker=1", "tmpl_upd_mode=disabled"], check=True)
    for record in records:
        if "repaired_path" in record:
            path = ROOT/record["repaired_path"]
            record["physics_success"] = Path(str(path).replace("/grasp_data/", "/succ_grasp/")).exists()
    (args.output/"repair_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__": main()
