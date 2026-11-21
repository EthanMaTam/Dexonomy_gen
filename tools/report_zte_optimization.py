"""比较冻结候选的独立物体 A/B 结果，发布达到预设改善条件的 ZTE 标注。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import mujoco

from dexonomy.util.file_util import load_yaml
from tools.prepare_zte_hand import ROOT, DESTINATION
from tools.select_zte_candidates import qualified_counts
from tools.sharpa_template_editor import _contact_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"output/zte_annotation_tuning")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-prefix", choices=["optimization", "rest_optimization"], default="optimization")
    args = parser.parse_args(); directory = args.directory.resolve()
    selection = json.loads((directory/"selection.json").read_text())
    model_hash = hashlib.sha256((DESTINATION/"right.xml").read_bytes()).hexdigest()
    assert model_hash == selection["model_sha256"]
    model = mujoco.MjModel.from_xml_path(str(DESTINATION/"right.xml"))
    jobs = {}; metrics = {}; qualified = {}
    for name in ("heldout_baseline", "heldout_optimized"):
        run = directory/name
        status = json.loads((run/"status.json").read_text())
        assert status["state"] == "finished" and status["reason"] == "all_objects_processed", status
        jobs[name] = json.loads((run/"job.json").read_text())
        assert jobs[name]["model_sha256"] == model_hash
        assert not set(jobs[name]["objects"]) & set(selection["screen_objects"])
        metrics[name] = {r["template"]: r for r in csv.DictReader((run/"summary.csv").open())}
        qualified[name] = qualified_counts(run, model)
    repair_both = all((directory/name/"connect_repair/repair_results.json").exists()
                      for name in ("heldout_baseline", "heldout_optimized"))
    assert jobs["heldout_baseline"]["objects"] == jobs["heldout_optimized"]["objects"]
    assert selection["selected_at_unix"] <= jobs["heldout_optimized"]["started_at"]
    for key in ("points", "scales", "workers", "gpu"):
        assert jobs["heldout_baseline"]["settings"][key] == jobs["heldout_optimized"]["settings"][key]
    for op in ("grasp", "eval"):
        paths = {name: directory/f"{name}/shards/0000/log/{op}_0/.hydra/config.yaml" for name in jobs}
        missing = [name for name, p in paths.items() if not p.exists()]
        if missing:
            assert op == "eval"
            assert all(sum(int(r["grasps"]) for r in metrics[name].values()) == 0 for name in missing)
            assert all(sum(int(r["successful"]) for r in metrics[name].values()) == 0 for name in missing)
            continue  # No refined candidates means evaluation was legitimately skipped.
        a = load_yaml(str(paths["heldout_baseline"]))
        b = load_yaml(str(paths["heldout_optimized"]))
        assert a["op"] == b["op"] and a["hand"] == b["hand"]
    rows = []
    for choice in selection["selection"]:
        name = choice["template"]
        old_path = directory/"heldout_baseline/raw_anno_snapshot"/(name+".yaml")
        new_path = directory/"heldout_optimized/raw_anno_snapshot"/(name+".yaml")
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == choice["baseline_sha256"]
        assert hashlib.sha256(new_path.read_bytes()).hexdigest() == choice["selected_sha256"]
        old, new = load_yaml(str(old_path)), load_yaml(str(new_path))
        assert list(old["contact"]) == list(new["contact"])
        assert all(len(_contact_entries(old["contact"][b])) == len(_contact_entries(new["contact"][b])) for b in old["contact"])
        assert old.get("required_cbody") == new.get("required_cbody")
        initial = int(metrics["heldout_baseline"][name]["initializations"])
        assert initial == int(metrics["heldout_optimized"][name]["initializations"]) == 144
        old_count = qualified["heldout_baseline"].get(name, 0)
        new_count = qualified["heldout_optimized"].get(name, 0)
        accepted = choice["candidate"] != "baseline" and choice["screen_qualified"] > 0 and new_count-old_count >= 3
        rows.append({**choice, "heldout_initializations": initial, "heldout_baseline_qualified": old_count,
                     "heldout_optimized_qualified": new_count,
                     "heldout_baseline_standard": int(metrics["heldout_baseline"][name]["successful"]),
                     "heldout_optimized_standard": int(metrics["heldout_optimized"][name]["successful"]),
                     "accepted": accepted})
    if args.apply:
        archive = DESTINATION/"annotation_history/2026-11-21_before_optimization"
        archive.mkdir(parents=True, exist_ok=True)
        # Archive the original reports before any current-library summary is replaced.
        for filename in ("taxonomy_validation.md", "taxonomy_validation.csv", "validation_summary.json",
                         "taxonomy_provenance.json", "taxonomy_metadata.json", "cross_validation.md", "cross_validation.csv"):
            target = archive/filename
            if not target.exists(): shutil.copy2(DESTINATION/filename, target)
        provenance_path = DESTINATION/"taxonomy_provenance.json"
        provenance = {r["template"]: r for r in json.loads(provenance_path.read_text())}
        for row in rows:
            name = row["template"]
            source = directory/"heldout_baseline/raw_anno_snapshot"/(name+".yaml")
            archive_file = archive/(name+".yaml")
            if archive_file.exists(): assert archive_file.read_bytes() == source.read_bytes()
            else: shutil.copy2(source, archive_file)
            if not row["accepted"]: continue
            current = DESTINATION/"raw_anno"/(name+".yaml")
            selected = directory/"heldout_optimized/raw_anno_snapshot"/(name+".yaml")
            assert current.read_bytes() in (source.read_bytes(), selected.read_bytes())
            shutil.copy2(selected, current)
            provenance[name].update({"status": "接触重定位优化并通过独立物体 A/B 验证，分类语义待人工审阅",
                                     "sha256": row["selected_sha256"],
                                     "last_optimization": {"candidate": row["candidate"], "baseline_sha256": row["baseline_sha256"],
                                                           "heldout_baseline_qualified": row["heldout_baseline_qualified"],
                                                           "heldout_optimized_qualified": row["heldout_optimized_qualified"]}})
        provenance_path.write_text(json.dumps(list(provenance.values()), ensure_ascii=False, indent=2)+"\n")
    result = {"model_sha256": model_hash, "candidate_count": selection["candidate_count"],
              "selection_frozen_before_optimized_validation": True, "object_sets_disjoint": True,
              "publication_rule": "调参集至少一个闭环合格成功样本，独立六物体至少增加三个闭环合格成功样本；单种子工程检查，不是统计显著性结论",
              "model_torque_and_global_thresholds_changed": False,
              "same_closed_loop_repair_rule_applied_to_both_groups": repair_both,
              "accepted_templates": [r["template"] for r in rows if r["accepted"]],
              "selection": rows}
    text = "# ZTE 缺陷改进与独立物体验证\n\n"
    text += ("候选保留接触指节、接触点数及必要分组，模型、力矩和全局验收阈值不变。"
             "比较指腹重定位、三种虚拟球半径和部分类型的三种虚拟圆柱半径；同一类型的候选使用相同调参物体。"
             "候选仅按调参结果选择并冻结，随后在另外六个不重叠物体上 A/B 验证。\n\n"
             "发布条件：调参集至少一个连接误差合格样本，独立验证至少增加三个合格样本。"
             "该条件是工程筛选规则，样本少、单随机种子，不能作为统计显著性或硬件可用性结论。\n\n"
             "合格指原引擎物理评估成功且最终姿态连接误差 ≤0.2 mm。已有逐例动态审计的样本也须在采样中满足同一限值。\n\n"
             "| 类型 | 冻结方案 | 调参合格/候选 | 独立验证原始/优化合格数 | 发布 |\n"
             "| --- | --- | ---: | ---: | --- |\n")
    if repair_both:
        text = text.replace("合格指原引擎物理评估成功且最终姿态连接误差 ≤0.2 mm。",
                            "两组使用相同的闭环修复规则：超限样本仅调整被动坐标，重新通过原碰撞、接触、QP、预抓取和外力评估后才计入；动态逐样本审计中超过 0.2 mm 的样本也排除。合格指物理评估成功、最终姿态及已审计动态采样的连接误差均 ≤0.2 mm。")
    for row in rows:
        text += f"| {row['template']} | {row['candidate']} | {row['screen_qualified']}/{row['screen_initializations']} | {row['heldout_baseline_qualified']}/{row['heldout_optimized_qualified']} | {'是' if row['accepted'] else '保留原标注'} |\n"
    total_initial = sum(r["heldout_initializations"] for r in rows)
    old = sum(r["heldout_baseline_qualified"] for r in rows)
    new = sum(r["heldout_optimized_qualified"] for r in rows)
    text += f"\n{len(rows)} 类独立验证合计：{old}/{total_initial} → {new}/{total_initial}；此数含未达到发布条件的试验候选，不是当前库全部采用后的成功率。\n"
    published = sum(r["heldout_optimized_qualified"] if r["accepted"] else r["heldout_baseline_qualified"] for r in rows)
    result["published_group_baseline_qualified"] = old
    result["published_group_optimized_qualified"] = published
    text += f"\n仅替换达到发布条件的类别、其余保留原标注时，验证结果的组合统计为 {old}/{total_initial} → {published}/{total_initial}。\n"
    if args.report_prefix == "rest_optimization":
        text = text.replace("比较指腹重定位、三种虚拟球半径和部分类型的三种虚拟圆柱半径；同一类型的候选使用相同调参物体。",
                            "比较未参与接触手指的收拢和侧摆姿态，保持参与接触指节的局部接触位置与法向；同一类型的候选使用相同的六个调参物体。")
    text += "\n原标注与报告备份在 annotation_history/2026-11-21_before_optimization/。旧全库表是优化前快照，不能当作当前库的全库分数。\n"
    (directory/"optimization_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (directory/"optimization_report.md").write_text(text)
    if args.apply:
        (DESTINATION/(args.report_prefix+"_provenance.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        (DESTINATION/(args.report_prefix+"_report.md")).write_text(text)
        old_report = DESTINATION/"taxonomy_validation.md"
        note = f"> 本表对应优化前的完整标注快照。当前缺陷改进与独立验证结果请查看 [优化报告]({args.report_prefix}_report.md)。\n\n"
        content = old_report.read_text()
        if note not in content:
            old_report.write_text(content.replace("# ZTE 右手全分类验收\n\n", "# ZTE 右手全分类验收\n\n"+note, 1))
    print(text)


if __name__ == "__main__": main()
