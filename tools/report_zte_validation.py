"""汇总 ZTE 全分类生成、闭环误差和已有三种手构型的同条件参考结果。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from dexonomy.util.file_util import load_yaml
from tools.closed_loop_hand import connect_errors
from tools.prepare_zte_hand import ROOT, DESTINATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"output/zte_hand_validation")
    parser.add_argument("--loop-limit", type=float, default=.0002,
                        help="Final grasp anchor-error engineering limit in metres; not a hardware specification")
    args = parser.parse_args()
    if args.loop_limit <= 0:
        parser.error("loop-limit must be positive")
    run = args.run.resolve()
    job = json.loads((run/"job.json").read_text())
    status = json.loads((run/"status.json").read_text())
    assert status["state"] == "finished" and status["reason"] == "all_objects_processed", status
    model_hash = hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest()
    assert job["model_sha256"] == model_hash
    geometry = {r["name"]: r for r in json.loads((run/"asset_checks.json").read_text())["templates"]}
    model_checks = json.loads((run/"model_checks.json").read_text())
    assert model_checks["model_sha256"] == model_hash
    paths = sorted((DESTINATION/"raw_anno").glob("*.yaml"))
    assert len(paths) == 34 and set(geometry) == {p.stem for p in paths}
    counts = {r["template"]: r for r in csv.DictReader((run/"summary.csv").open())}
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml"))
    state = mujoco.MjData(model)
    qualified = {p.stem: 0 for p in paths}
    maxima = {p.stem: 0. for p in paths}
    observed = {p.stem: 0 for p in paths}
    samples = []
    for p in sorted(run.glob("shards/*/succ_grasp/**/*.npy")):
        root = next(a for a in p.parents if a.name == "succ_grasp")
        name = p.relative_to(root).parts[0]
        data = np.load(p, allow_pickle=True).item()
        error = 0.
        for qpos in data["grasp_qpos"]:
            assert qpos.shape == (model.nq+7,) and np.isfinite(qpos).all(), p
            state.qpos[:] = qpos[7:]
            mujoco.mj_kinematics(model, state)
            error = max(error, float(np.linalg.norm(connect_errors(model, state), axis=1).max()))
        accepted = error <= args.loop_limit
        observed[name] += 1
        qualified[name] += int(accepted)
        maxima[name] = max(maxima[name], error)
        samples.append({"path": str(p.relative_to(run)), "template": name,
                        "max_connect_error_m": error, "closed_loop_accepted": accepted})
    rows = []
    for path in sorted(paths, key=lambda p: (int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 1000)):
        assert path.read_bytes() == (run/"raw_anno_snapshot"/path.name).read_bytes(), path
        name = path.stem
        c = counts[name]
        assert observed[name] == int(c["successful"]), name
        rows.append({"template": name, "contact_points": geometry[name]["contacts"],
                     "initializations": int(c["initializations"]), "refined": int(c["grasps"]),
                     "simulation_successful": int(c["successful"]), "closed_loop_accepted": qualified[name],
                     "max_grasp_connect_error_m": maxima[name],
                     "annotation_connect_error_m": geometry[name]["max_connect_error_m"],
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path, records in [(DESTINATION/"taxonomy_validation.csv", rows), (run/"closed_loop_samples.csv", samples)]:
        if records:
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
    totals = {k: sum(r[k] for r in rows) for k in ["initializations", "refined", "simulation_successful", "closed_loop_accepted"]}
    successful_categories = sum(r["closed_loop_accepted"] > 0 for r in rows if r["template"].split("_")[0].isdigit())
    replay = json.loads((run/"connect_replay.json").read_text())
    assert replay["model_sha256"] == model_hash
    (DESTINATION/"model_validation.json").write_text(json.dumps(model_checks, ensure_ascii=False, indent=2)+"\n")
    (DESTINATION/"connect_replay.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2)+"\n")
    report = {"model_sha256": model_hash, "templates": 34, "numbered_categories": 33,
              "numbered_categories_with_accepted_samples": successful_categories,
              "needs_improvement": [r["template"] for r in rows if not r["closed_loop_accepted"]],
              "objects": len(job["objects"]), "scales": job["settings"]["scales"],
              "points": job["settings"]["points"], "totals": totals,
              "final_grasp_connect_limit_m": args.loop_limit,
              "max_successful_grasp_connect_error_m": max(maxima.values()),
              "replayed_classes": replay["count"], "replay_max_connect_error_m": replay["max_connect_error_m"],
              "all_replayed_physics_successful": replay["all_replayed_physics_successful"],
              "replay_sampled_states_within_final_grasp_limit": replay["max_connect_error_m"] <= args.loop_limit,
              "annotations_match_snapshot": True,
              "geometry_check_max_connect_error_m": max(r["max_connect_error_m"] for r in geometry.values()),
              "source_kinematics_verified_poses": model_checks["source_kinematics_verified_poses"]}
    text = ("# ZTE 右手全分类验收\n\n"
            f"{len(job['objects'])} 个物体、{len(job['settings']['scales'])} 个尺度、{job['settings']['points']} 点采样，"
            "沿用原抓取、碰撞、接触、QP 及六方向外力评估阈值。\n\n"
            f"34 份标注通过关节、表面、法向及闭环检查。{totals['initializations']} 个初始候选、"
            f"{totals['refined']} 个有效抓取、{totals['simulation_successful']} 个标准仿真成功样本；"
            f"其中 {totals['closed_loop_accepted']} 个最终抓取姿态的连接误差不超过 {1000*args.loop_limit:g} mm。"
            f"编号类型 {successful_categories}/33 类有满足该附加条件的成功样本。\n\n"
            "连接误差限是本次集成检查的工程限值，未经硬件公差标定；不是实物成功率或全轨迹误差上界。"
            "若逐类没有成功样本，保留标注并标记需改进，不将几何通过等同于抓取可用。\n\n"
            f"分类型复跑了 {replay['count']} 个成功抓取，六方向物理评估均再次通过。"
            f"在 step_sim 返回后采样的最大连接误差为 {1000*replay['max_connect_error_m']:.4f} mm，"
            "并未逐积分步采样。完整采样记录见运行目录的 connect_replay.json。\n\n"
            "| 类型 | 接触数 | 候选 | 有效抓取 | 标准仿真成功 | 连接误差合格 | 最大成功姿态连接误差 mm |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in rows:
        text += (f"| {row['template']} | {row['contact_points']} | {row['initializations']} | {row['refined']} | "
                 f"{row['simulation_successful']} | {row['closed_loop_accepted']} | {1000*row['max_grasp_connect_error_m']:.4f} |\n")
    text += "\n待改进（本次无连接误差合格的成功样本）：" + "、".join(report["needs_improvement"]) + "。\n"
    text += "\n全部标注由 O30 程序辅助迁移，保留所需接触分组；手势分类语义仍需人工审核。\n"
    (DESTINATION/"taxonomy_validation.md").write_text(text)
    (DESTINATION/"validation_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (run/"verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    references = {"Shadow": ROOT/"output/shadow_cross_validation", "Sharpa": ROOT/"output/sharpa_o30_comparison",
                  "O30_历史基线": ROOT/"output/o30_taxonomy_verified", "ZTE": run}
    if not all((p/"summary.csv").exists() for p in references.values()):
        return
    metrics = {name: {r["template"]: r for r in csv.DictReader((p/"summary.csv").open())} for name, p in references.items()}
    for name, path in references.items():
        ref_job = json.loads((path/"job.json").read_text())
        assert ref_job["objects"] == job["objects"]
        for key in ("points", "scales"):
            assert ref_job["settings"][key] == job["settings"][key]
        for op in ("grasp", "eval"):
            assert load_yaml(str(path/f"shards/0000/log/{op}_0/.hydra/config.yaml"))["op"] == load_yaml(str(run/f"shards/0000/log/{op}_0/.hydra/config.yaml"))["op"]
    common = sorted(set.intersection(*(set(v) for v in metrics.values())), key=lambda n: int(n.split("_")[0]))
    assert len(common) == 26
    comparison = []
    for name, values in metrics.items():
        initial = sum(int(values[n]["initializations"]) for n in common)
        successful = sum(int(values[n]["successful"]) for n in common)
        comparison.append({"hand": name, "common_categories": len(common), "initializations": initial,
                           "refined": sum(int(values[n]["grasps"]) for n in common),
                           "successful": successful, "success_percent": 100*successful/initial})
    text = "# ZTE 与现有手构型交叉验证\n\n共有 26 类、相同六物体和三尺度、512 点采样；细化与评估配置一致。\n\n"
    text += "| 构型 | 候选 | 有效抓取 | 标准仿真成功 | 成功率 |\n| --- | ---: | ---: | ---: | ---: |\n"
    for row in comparison:
        text += f"| {row['hand']} | {row['initializations']} | {row['refined']} | {row['successful']} | {row['success_percent']:.2f}% |\n"
    text += ("\n这里采用原引擎成功口径。ZTE 的额外连接误差合格数另见 taxonomy_validation.md。"
             "ZTE 为 40 坐标、15 执行器及 10 组连接约束；O30 为 20 个独立执行器；两者均采用 ±1 Nm 仿真假设。"
             "ZTE 保留来源的阻尼及电枢惯量；控制、几何和接触要求不完全相同，比较不能单独归因于标注。"
             "Shadow 为 22 关节、18 执行器及 ±3 Nm；Sharpa 未设力矩上限。O30 行是优化前历史全库，"
             "ZTE 迁移来源含 O30 后来的四类优化，不能当作仅改变手部结构的对照实验。\n")
    (DESTINATION/"cross_validation.md").write_text(text)
    with (DESTINATION/"cross_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)


if __name__ == "__main__":
    main()
