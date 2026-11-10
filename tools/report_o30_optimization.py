"""报告 O30 标注优化的独立物体 A/B 结果，并可发布通过验证的候选。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from dexonomy.util.file_util import load_yaml
from tools.optimize_o30_annotations import ROOT
from tools.sharpa_template_editor import _contact_entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=ROOT/"output/o30_annotation_tuning")
    parser.add_argument("--baseline-run",type=Path,default=ROOT/"output/o30_taxonomy_verified")
    parser.add_argument("--apply",action="store_true",help="发布独立验证中至少增加三个成功样本的候选")
    args=parser.parse_args();directory=args.directory
    selected=json.loads((directory/"selection.json").read_text())
    jobs={name:json.loads((directory/name/"job.json").read_text())
          for name in ["train","heldout_baseline","heldout_optimized"]}
    assert not set(jobs["train"]["objects"])&set(jobs["heldout_baseline"]["objects"])
    assert jobs["heldout_baseline"]["objects"]==jobs["heldout_optimized"]["objects"]
    for name in jobs:
        status=json.loads((directory/name/"status.json").read_text())
        assert status["state"]=="finished" and status["reason"]=="all_objects_processed"
        for key in ["points","scales","workers","gpu"]:
            assert jobs[name]["settings"][key]==jobs["train"]["settings"][key]
    assert selected["selected_at_unix"]<=jobs["heldout_optimized"]["started_at"]
    for op in ["grasp","eval"]:
        configs=[load_yaml(str(directory/name/f"shards/0000/log/{op}_0/.hydra/config.yaml"))["op"]
                 for name in jobs]
        assert all(config==configs[0] for config in configs)
    metrics={name:{r["template"]:r for r in csv.DictReader((directory/name/"summary.csv").open())}
             for name in ["heldout_baseline","heldout_optimized"]}
    hand_dir=ROOT/"assets/hand/linker_o30_right"
    model_hash=hashlib.sha256((hand_dir/"right.xml").read_bytes()).hexdigest()
    assert model_hash==json.loads((args.baseline_run/"verification.json").read_text())["model_sha256"]
    for job in jobs.values():
        if "model_sha256" in job:
            assert model_hash==job["model_sha256"], "Hand model differs between runs"
    rows=[]
    for choice in selected["selection"]:
        name=choice["template"]
        old=metrics["heldout_baseline"][name];new=metrics["heldout_optimized"][name]
        assert int(old["initializations"])==int(new["initializations"])==144
        old_file=directory/"heldout_baseline/raw_anno_snapshot"/(name+".yaml")
        new_file=directory/"heldout_optimized/raw_anno_snapshot"/(name+".yaml")
        assert hashlib.sha256(new_file.read_bytes()).hexdigest()==choice["selected_sha256"]
        old_data,new_data=load_yaml(str(old_file)),load_yaml(str(new_file))
        assert list(old_data["contact"])==list(new_data["contact"])
        for body in old_data["contact"]:
            assert len(_contact_entries(old_data["contact"][body]))==len(_contact_entries(new_data["contact"][body]))
        assert old_data.get("required_cbody")==new_data.get("required_cbody")
        accepted=(int(new["successful"])-int(old["successful"])>=3
                  and choice["train_selected_success"]>choice["train_baseline_success"])
        if args.apply and accepted:
            current=hand_dir/"raw_anno"/(name+".yaml")
            # A second invocation can report the published version safely.
            assert current.read_bytes() in [old_file.read_bytes(),new_file.read_bytes()],current
            archive=hand_dir/"annotation_history/2026-11-10_before_optimization"/(name+".yaml")
            archive.parent.mkdir(parents=True,exist_ok=True)
            if archive.exists():assert archive.read_bytes()==old_file.read_bytes()
            else:shutil.copy2(old_file,archive)
            shutil.copy2(new_file,current)
        rows.append({**choice,"heldout_initializations":144,
                     "heldout_baseline_success":int(old["successful"]),
                     "heldout_optimized_success":int(new["successful"]),
                     "heldout_baseline_percent":100*int(old["successful"])/144,
                     "heldout_optimized_percent":100*int(new["successful"])/144,
                     "accepted":accepted})
    original=sum(r["heldout_baseline_success"] for r in rows)
    optimized=sum(r["heldout_optimized_success"] for r in rows)
    result={"candidates_evaluated":17,"tuning_objects":6,"heldout_objects":6,
            "object_sets_disjoint":True,"selection_frozen_before_validation":True,
            "model_sha256":model_hash,"thresholds_changed":False,
            "heldout_baseline_success":original,"heldout_optimized_success":optimized,
            "heldout_initializations":576,"accepted_templates":[r["template"] for r in rows if r["accepted"]],
            "selection":rows}
    (directory/"optimization_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    text="# O30 特定标注优化与独立验证\n\n"
    text+="在原六个物体上评估 17 份候选，保留原标注及接触指节集合。候选仅按调参集成功数选择并冻结，再对另外六个不重叠物体做 A/B 验证；每类 144 个候选、三种尺度、512 点采样。\n\n"
    text+="模型、力矩限制、接触要求和全局评估阈值均未改变。只有独立验证比原标注至少增加三个成功样本，且调参集也改善的候选才发布。此处是小规模、单随机种子的工程验证，不是统计显著性结论。\n\n"
    text+="| 类型 | 选中方案 | 调参集原始/优化成功数 | 独立验证原始/优化成功数 | 独立验证原始/优化成功率 | 发布 |\n| --- | --- | ---: | ---: | --- | --- |\n"
    for row in rows:
        text+=f"| {row['template']} | {row['candidate']} | {row['train_baseline_success']}/{row['train_selected_success']} | {row['heldout_baseline_success']}/{row['heldout_optimized_success']} | {row['heldout_baseline_percent']:.2f}%/{row['heldout_optimized_percent']:.2f}% | {'是' if row['accepted'] else '保留原标注'} |\n"
    text+=f"\n四类独立验证合计：{original}/576（{100*original/576:.2f}%）→ {optimized}/576（{100*optimized/576:.2f}%）。\n\n"
    text+="优化改变的是接触位置、对向或张开尺度，未减少所需接触指节。Tripod 的宽姿态来自已有同样三指拓扑的 fingertip_large；其余候选为指腹中心投影或带自碰撞惩罚的虚拟球拟合。全部属于自动辅助优化，分类语义仍需人工审阅。\n\n原标注备份于 `annotation_history/2026-11-10_before_optimization/`；原始全分类报告对应优化前的快照，不应直接当作新库的整体成功率。\n"
    (directory/"optimization_report.md").write_text(text)
    if args.apply:
        (hand_dir/"optimization_provenance.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        (hand_dir/"optimization_report.md").write_text(text)
    print(text)


if __name__=="__main__":main()
