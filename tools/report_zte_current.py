"""汇总当前 ZTE 库的分阶段验收证据，不混算不同实验的全库成功率。"""

import argparse
import csv
import hashlib
import json
import re

import numpy as np

from dexonomy.util.closed_loop_util import project_free_hand_grasp

from tools.prepare_zte_hand import ROOT, DESTINATION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-run", default="output/zte_grasp_repair_verified")
    args = parser.parse_args()
    if (DESTINATION/"retarget_report.json").exists():
        from tools.report_shadow_retarget import write_report
        run=ROOT/"output/shadow_retarget/zte_generation_smoke"
        write_report("zte_hand",run if (run/"summary.csv").exists() else None)
        return
    archive = DESTINATION/"annotation_history/2026-11-21_before_optimization"
    baseline = {r["template"]: r for r in csv.DictReader((archive/"taxonomy_validation.csv").open())}
    geometry = {r["name"]: r for r in json.loads((ROOT/"output/zte_annotation_tuning/published_asset_checks.json").read_text())["templates"]}
    model_hash = hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest()
    evidence = {}; baseline_evidence = {}
    for filename in ("optimization_provenance.json", "rest_optimization_provenance.json"):
        path = DESTINATION/filename
        if not path.exists(): continue
        experiment = json.loads(path.read_text())
        assert experiment["model_sha256"] == model_hash
        for row in experiment["selection"]:
            if row["accepted"]: evidence[row["template"]] = {**row, "report": filename.replace("_provenance.json", "_report.md")}
            elif row["heldout_baseline_qualified"] > 0:
                old = baseline_evidence.get(row["template"])
                if old is None or row["heldout_baseline_qualified"] > old["heldout_baseline_qualified"]:
                    baseline_evidence[row["template"]] = {**row, "report": filename.replace("_provenance.json", "_report.md")}
    records = []
    for path in sorted((DESTINATION/"raw_anno").glob("*.yaml")):
        name = path.stem; old = baseline[name]; latest = evidence.get(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == (latest["selected_sha256"] if latest else old["sha256"]), path
        count = latest["heldout_optimized_qualified"] if latest else int(old["closed_loop_accepted"])
        retained = baseline_evidence.get(name) if latest is None else None
        if retained is not None:
            assert digest == retained["baseline_sha256"]
            count = max(count, retained["heldout_baseline_qualified"])
        records.append({"template": name, "sha256": digest, "geometry_passed": True,
                        "max_annotation_connect_error_m": geometry[name]["max_connect_error_m"],
                        "has_qualified_success": count > 0, "qualified_samples_in_supporting_run": count,
                        "supporting_experiment": latest["report"] if latest else retained["report"] if retained else "首次全库验证（该标注未修改）",
                        "annotation_changed": latest is not None})
    repair_run = ROOT/args.repair_run
    repairs = json.loads((repair_run/"repair_results.json").read_text())
    assert len(repairs) == 2 and all(r.get("physics_success") for r in repairs)
    replay = json.loads((repair_run/"connect_replay.json").read_text())
    assert replay["model_sha256"] == model_hash and replay["all_replayed_physics_successful"]
    guard_dir = ROOT/"output/zte_connect_guard_minimal/shards/0000"
    assert len(list((guard_dir/"succ_grasp").rglob("*.npy"))) == 2
    no_op_count = 0
    original_run = ROOT/"output/zte_hand_validation"
    for row in csv.DictReader((original_run/"closed_loop_samples.csv").open()):
        if row["closed_loop_accepted"] != "True": continue
        qpos = np.load(original_run/row["path"], allow_pickle=True).item()["grasp_qpos"][0]
        projected, changed = project_free_hand_grasp(DESTINATION/"right.xml", qpos, .0002)
        assert not changed and np.array_equal(projected, qpos)
        no_op_count += 1
    assert no_op_count == 342
    dynamic_dir = ROOT/"output/zte_dynamic_guard_regression/shards/0000/log/eval_0/connect_checks"
    dynamic_checks = [{"path": str(p.relative_to(ROOT)), **json.loads(p.read_text())} for p in dynamic_dir.rglob("*.json")]
    assert len(dynamic_checks) == 5
    assert all(r["physics_success"] for r in dynamic_checks)
    assert sum(r["accepted"] for r in dynamic_checks) == 4
    summary = {"model_sha256": model_hash,
               "hand_config_sha256": hashlib.sha256((ROOT/"dexonomy/config/hand/zte_hand.yaml").read_bytes()).hexdigest(),
               "numbered_categories_with_qualified_samples": sum(r["has_qualified_success"] for r in records if r["template"].split("_")[0].isdigit()),
               "numbered_categories": 33, "templates": len(records),
               "published_templates": list(evidence),
               "unchanged_templates_with_additional_independent_evidence": list(baseline_evidence),
               "needs_improvement": [r["template"] for r in records if not r["has_qualified_success"]],
               "coverage_method": "未修改标注沿用首次全库证据；优化标注使用冻结候选的独立物体证据；未合并计算新全库成功率",
               "repaired_original_outliers": len(repairs),
               "repaired_max_final_connect_error_m": max(r["new_connect_error_m"] for r in repairs),
               "repair_replay_max_connect_error_m": replay["max_connect_error_m"],
               "generator_regression_original_fixtures_passed": 2,
               "original_qualified_poses_unchanged": no_op_count,
               "dynamic_guard_regression_accepted": 4,
               "dynamic_guard_regression_rejected_for_loop_error": 1,
               "legacy_classes_not_fully_reaudited_by_new_dynamic_gate": 26,
               "records": records}
    (DESTINATION/"current_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    (DESTINATION/"closure_repair_provenance.json").write_text(json.dumps({"repairs": repairs, "replay": replay}, ensure_ascii=False, indent=2)+"\n")
    (DESTINATION/"dynamic_guard_regression.json").write_text(json.dumps(dynamic_checks, ensure_ascii=False, indent=2)+"\n")
    text = ("# ZTE 当前标注库验收索引\n\n"
            f"当前 34 份标注的几何与闭环检查通过，编号类型中 {summary['numbered_categories_with_qualified_samples']}/33 类已有成功证据，各阶段口径见下。\n\n"
            "本表按类别列出支撑证据：未修改的标注保留首次全库验证，优化的标注采用独立物体验证。"
            "这些来自不同实验，不将其合并宣称为一次新全库成功率；首次全库报告保留为历史快照。\n\n"
            "原 26 类的证据按当时的物理评估与最终姿态闭环口径保留，尚未逐例按新增动态保护重算；新发布的 Lateral 已逐例动态审计。\n\n"
            "| 类型 | 支撑实验合格样本 | 状态 | 证据 |\n| --- | ---: | --- | --- |\n")
    for row in sorted(records, key=lambda r: int(r["template"].split("_")[0]) if r["template"].split("_")[0].isdigit() else 1000):
        evidence_text = (f"[独立验证：{'新标注' if row['annotation_changed'] else '原标注保留'}]({row['supporting_experiment']})" if row["template"] in evidence or row["template"] in baseline_evidence else "首次全库，标注未修改")
        text += f"| {row['template']} | {row['qualified_samples_in_supporting_run']} | {'已有合格成功样本' if row['has_qualified_success'] else '仍需改进'} | {evidence_text} |\n"
    text += "\n仍需改进："+"、".join(summary["needs_improvement"])+"。\n\n"
    text += ("两个原闭环超限样本已采用最小必要的被动关节修正，原抓取电机目标及基座不变，"
             "重新通过碰撞、接触、QP、预抓取和六方向外力检查。"
             f"最终连接误差约 {1000*summary['repaired_max_final_connect_error_m']:.3f} mm，"
             f"两例动态复跑采样的最大误差为 {1000*summary['repair_replay_max_connect_error_m']:.4f} mm。\n\n"
             "ZTE 配置已启用生成器保护：超限时先投影、再运行原检查，不可修复或重新检查不通过则拒绝。"
             "此前 342 个合格成功姿态经新投影函数检查均逐坐标保持不变；两个原始初始化夹具经实际生成器和评估回归均通过。\n\n"
             "评估阶段新增动态保护：每次 step_sim 返回后采样连接误差，超过 0.2 mm 则拒绝。五个真实夹具回归中，四个通过、一个虽通过外力检查但因约 0.206 mm 的动态误差被正确拒绝。此采样不等于逐积分步的误差上界。\n\n"
             "来源模型、传动拓扑、力矩和全局抓取阈值未改变；这是小规模仿真验证，分类语义和实物表现仍待确认。\n")
    (DESTINATION/"current_validation.md").write_text(text)
    old_table = DESTINATION/"taxonomy_validation.md"
    note = (f"> 当前分阶段证据已更新：**{summary['numbered_categories_with_qualified_samples']}/33 类已有成功证据**。"
            "详见 [当前验收索引](current_validation.md)；下表为首次全库历史快照，未按新增动态保护全量重算。\n\n")
    content = re.sub(r"> 当前分阶段证据已更新：[^\n]*\n\n", "", old_table.read_text())
    old_table.write_text(content.replace("# ZTE 右手全分类验收\n\n", "# ZTE 右手全分类验收\n\n"+note, 1))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
