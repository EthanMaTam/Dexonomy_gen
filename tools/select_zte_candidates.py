"""仅使用调参结果选择并冻结 ZTE 候选，独立验证不参与选择。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from tools.prepare_zte_hand import ROOT, DESTINATION
from tools.closed_loop_hand import connect_errors


def qualified_counts(run, model):
    result = {}; state = mujoco.MjData(model); rejected = set()
    replay = run/"connect_replay.json"
    dynamic_original = ({(ROOT/r["path"]).resolve(): r for r in json.loads(replay.read_text())["records"]}
                        if replay.exists() else {})
    for path in run.glob("shards/*/succ_grasp/**/*.npy"):
        root = next(p for p in path.parents if p.name == "succ_grasp")
        name = path.relative_to(root).parts[0]
        data = np.load(path, allow_pickle=True).item()
        error = 0.
        for qpos in data["grasp_qpos"]:
            state.qpos[:] = qpos[7:]; mujoco.mj_kinematics(model, state)
            error = max(error, float(np.linalg.norm(connect_errors(model, state), axis=1).max()))
        dynamic_ok = path.resolve() not in dynamic_original or dynamic_original[path.resolve()]["max_connect_error_m"] <= .0002
        result[name] = result.get(name, 0)+int(error <= .0002 and dynamic_ok)
        if error > .0002: rejected.add(path.resolve())
    repairs = run/"connect_repair/repair_results.json"
    if repairs.exists():
        replay_path = repairs.parent/"connect_replay.json"
        dynamic = ({r["path"]: r for r in json.loads(replay_path.read_text())["records"]}
                   if replay_path.exists() else {})
        for record in json.loads(repairs.read_text()):
            if not record.get("physics_success"): continue
            if record["repaired_path"] in dynamic and dynamic[record["repaired_path"]]["max_connect_error_m"] > .0002:
                continue
            source = ROOT/record["source"]
            assert source.resolve() in rejected
            assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
            corrected = ROOT/record["repaired_path"]
            assert Path(str(corrected).replace("/grasp_data/", "/succ_grasp/")).exists()
            qpos = np.load(corrected, allow_pickle=True).item()["grasp_qpos"][0]
            state.qpos[:] = qpos[7:]; mujoco.mj_kinematics(model, state)
            assert float(np.linalg.norm(connect_errors(model, state), axis=1).max()) <= .0002
            result[record["template"]] = result.get(record["template"], 0)+1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"output/zte_annotation_tuning")
    parser.add_argument("--rest", action="store_true", help="Select the unused-finger posture experiment")
    parser.add_argument("--baseline-run", type=Path, default=ROOT/"output/zte_hand_validation")
    args = parser.parse_args(); directory = args.directory.resolve()
    output = directory/"selected_annotations"
    if output.exists(): parser.error("选择已冻结，不覆盖已有选择")
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml"))
    baseline_run = args.baseline_run.resolve()
    baseline_job = json.loads((baseline_run/"job.json").read_text())
    baseline_metrics = {r["template"]: r for r in csv.DictReader((baseline_run/"summary.csv").open())}
    observations = {}; jobs = []
    experiments = [directory, directory/"spread"] if args.rest else [directory, directory/"cylinder", directory/"additional_distal"]
    for experiment in experiments:
        run = experiment/("train" if args.rest else "screen")
        status = json.loads((run/"status.json").read_text())
        assert status["state"] == "finished" and status["reason"] == "all_objects_processed", status
        job = json.loads((run/"job.json").read_text()); jobs.append(job)
        extra = experiment.name == "additional_distal"
        expected_objects = baseline_job["objects"] if args.rest else (baseline_job["objects"][2:6] if extra else baseline_job["objects"][:2])
        assert job["objects"] == expected_objects
        assert job["model_sha256"] == baseline_job["model_sha256"]
        qualified = qualified_counts(run, model)
        provenance = {r["candidate"]: r for r in json.loads((experiment/"candidate_provenance.json").read_text())}
        for row in csv.DictReader((run/"summary.csv").open()):
            name = row["template"]; origin = provenance[name]
            source = run/"raw_anno_snapshot"/(name+".yaml")
            assert hashlib.sha256(source.read_bytes()).hexdigest() == origin["sha256"]
            assert hashlib.sha256((baseline_run/"raw_anno_snapshot"/(origin["template"]+".yaml")).read_bytes()).hexdigest() == origin["baseline_sha256"]
            assert int(row["initializations"]) == (144 if args.rest else 96 if extra else 48)
            current = {
                "template": origin["template"],
                "candidate": name, "source": str(source.relative_to(ROOT)),
                "screen_initializations": int(row["initializations"]), "screen_qualified": qualified.get(name, 0),
                "screen_successful": int(row["successful"]), "screen_refined": int(row["grasps"]),
                "baseline_sha256": origin["baseline_sha256"], "selected_sha256": origin["sha256"],
            }
            if name in observations:
                old = observations[name]
                assert old["selected_sha256"] == current["selected_sha256"]
                for key in ("screen_initializations", "screen_qualified", "screen_successful", "screen_refined"):
                    old[key] += current[key]
            else:
                observations[name] = current
    assert jobs[0]["objects"] == jobs[1]["objects"]
    options = {}
    for row in observations.values():
        options.setdefault(row["template"], []).append(row)
    output.mkdir()
    records = []
    for name, choices in options.items():
        assert int(baseline_metrics[name]["successful"]) == 0
        assert len({r["screen_initializations"] for r in choices}) == 1
        chosen = sorted(choices, key=lambda r: (-r["screen_qualified"], -r["screen_successful"], -r["screen_refined"], r["candidate"]))[0]
        if chosen["screen_qualified"] == 0:
            source = baseline_run/"raw_anno_snapshot"/(name+".yaml")
            chosen = {**chosen, "candidate": "baseline", "source": str(source.relative_to(ROOT)),
                      "selected_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        source = ROOT/chosen["source"]
        shutil.copy2(source, output/(name+".yaml"))
        records.append({"template": name, **chosen})
    result = {"selected_at_unix": time.time(), "selection_uses_heldout_data": False,
              "screen_objects": baseline_job["objects"], "model_sha256": baseline_job["model_sha256"],
              "screen_protocol": ("原六物体比较未接触手指的收拢姿态，独立验证使用新物体" if args.rest else
                                  "除 Distal 用原六物体外，其余类型用前二物体；同一类型各候选物体集和候选数一致"),
              "candidate_count": sum(len(v) for v in options.values()), "selection": records}
    (directory/"selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
