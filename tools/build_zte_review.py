"""汇集已发布改进、保留原标注的独立验证及原缺陷修复样本，排除动态超限。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.closed_loop_hand import connect_errors
from tools.prepare_zte_hand import ROOT, DESTINATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"output/zte_improvement_review")
    parser.add_argument("--historical", action="store_true", help="Explicitly review pre-retarget historical experiments")
    args = parser.parse_args(); output = args.output.resolve()
    if (DESTINATION/"retarget_report.json").exists() and not args.historical:
        parser.error("当前库已从 Shadow 整手重建；本工具只汇集旧实验。历史审阅请加 --historical，新结果请查看当前生成任务的 succ_grasp。")
    if output.exists(): parser.error("查看输出已存在，请使用新目录")
    index = DESTINATION/"annotation_history/2026-11-27_before_shadow_retarget/current_validation.json" if args.historical else DESTINATION/"current_validation.json"
    current = {r["template"]: r for r in json.loads(index.read_text())["records"]}
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml")); state = mujoco.MjData(model)
    rows = []

    def collect(run, names, reason):
        replay = json.loads((run/"connect_replay.json").read_text())
        dynamic = {(ROOT/r["path"]).resolve(): r for r in replay["records"]}
        for p in run.glob("shards/*/succ_grasp/**/*.npy"):
            root = next(a for a in p.parents if a.name == "succ_grasp")
            relative = p.relative_to(root); name = relative.parts[0]
            if name not in names: continue
            assert p.resolve() in dynamic
            sample = dynamic[p.resolve()]
            if sample["max_connect_error_m"] > .0002: continue
            data = np.load(p, allow_pickle=True).item(); state.qpos[:] = data["grasp_qpos"][0, 7:]
            mujoco.mj_kinematics(model, state)
            error = float(np.linalg.norm(connect_errors(model, state), axis=1).max())
            if error > .0002: continue
            destination = output/"shards/0000/succ_grasp"/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert not destination.exists()
            destination.symlink_to(p.resolve())
            rows.append({"path": str(destination.relative_to(output)), "template": name,
                         "max_connect_error_m": error, "closed_loop_accepted": True,
                         "max_sampled_dynamic_connect_error_m": sample["max_connect_error_m"],
                         "source": str(p.relative_to(ROOT)), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "evidence": reason})

    for prefix, experiment in [("optimization", ROOT/"output/zte_annotation_tuning"),
                               ("rest_optimization", ROOT/"output/zte_rest_tuning")]:
        report = json.loads((DESTINATION/(prefix+"_provenance.json")).read_text())
        accepted = {r["template"] for r in report["selection"] if r["accepted"]}
        retained = {r["template"] for r in report["selection"] if not r["accepted"] and
                    r["heldout_baseline_qualified"] > 0 and current[r["template"]]["supporting_experiment"] == prefix+"_report.md"}
        if accepted:
            collect(experiment/"heldout_optimized", accepted, "已发布标注的独立验证")
            if (experiment/"heldout_optimized/connect_repair/connect_replay.json").exists():
                collect(experiment/"heldout_optimized/connect_repair", accepted, "已发布标注的闭环修复及逐例动态复跑")
        if retained: collect(experiment/"heldout_baseline", retained, "原标注保留，补充独立物体证据")
    collect(ROOT/"output/zte_grasp_repair_verified", {"9_Palmar_Pinch", "25_Lateral_Tripod"}, "原超限样本的修复与动态复跑")
    with (output/"closed_loop_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (output/"job.json").write_text(json.dumps({"review_only": True,
        "model_sha256": hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest(),
        "samples": len(rows), "note": "不同独立实验与修复样本的汇集，不能用于计算全库成功率"}, ensure_ascii=False, indent=2)+"\n")
    print(f"Prepared {len(rows)} individually audited samples in {output}")


if __name__ == "__main__": main()
