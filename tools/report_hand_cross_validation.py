"""汇总 Shadow、Sharpa 与 O30 在共有物体和抓取类型上的交叉验证。"""

import argparse
import csv
import json
from pathlib import Path

from dexonomy.util.file_util import load_yaml
from tools.sharpa_template_editor import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"output/three_hand_cross_validation")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    runs={"Shadow":ROOT/"output/shadow_cross_validation",
          "Sharpa":ROOT/"output/sharpa_o30_comparison",
          "O30_baseline":ROOT/"output/o30_taxonomy_verified"}
    jobs={name:json.loads((path/"job.json").read_text()) for name,path in runs.items()}
    for name,path in runs.items():
        status=json.loads((path/"status.json").read_text())
        assert status["state"]=="finished" and status["reason"]=="all_objects_processed"
        assert jobs[name]["objects"]==jobs["Shadow"]["objects"]
        for key in ["points","scales","workers","gpu"]:
            assert jobs[name]["settings"][key]==jobs["Shadow"]["settings"][key]
    for op in ["grasp","eval"]:
        configs=[load_yaml(str(path/f"shards/0000/log/{op}_0/.hydra/config.yaml"))["op"] for path in runs.values()]
        assert all(config==configs[0] for config in configs)
    summaries={name:{r["template"]:r for r in csv.DictReader((path/"summary.csv").open())}
               for name,path in runs.items()}
    common=sorted(set.intersection(*(set(rows) for rows in summaries.values())),key=lambda s:int(s.split("_")[0]))
    assert len(common)==26
    totals={}
    rows=[]
    for template in common:
        row={"template":template}
        for name in runs:
            data=summaries[name][template]
            assert int(data["initializations"])==144
            for old,new in [("initializations","initial"),("grasps","refined"),("successful","successful")]:
                row[f"{name}_{new}"]=int(data[old])
            row[f"{name}_success_percent"]=100*int(data["successful"])/int(data["initializations"])
        rows.append(row)
    for name in runs:
        d={key:sum(row[f"{name}_{key}"] for row in rows) for key in ["initial","refined","successful"]}
        d.update({"refinement_percent":100*d["refined"]/d["initial"],
                  "evaluation_percent":100*d["successful"]/max(1,d["refined"]),
                  "end_to_end_percent":100*d["successful"]/d["initial"],
                  "categories_with_success":sum(row[f"{name}_successful"]>0 for row in rows)})
        totals[name]=d
    with (args.output/"per_template.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]), lineterminator="\n");writer.writeheader();writer.writerows(rows)
    result={"common_categories":26,"objects":6,"scales":3,"points":512,"totals":totals,
            "note":"比较整套模型与标注配置，不能单独归因于标注。Sharpa 未设力矩限制；Shadow ±3 Nm，O30 ±1 Nm。"}
    (args.output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    text="# 三种手构型交叉验证\n\n相同 26 类、6 个物体、3 个尺度、512 点采样，抓取细化与仿真评估的全局参数一致。\n\n"
    text+="| 构型 | 初始候选 | 有效抓取 | 仿真成功 | 细化通过率 | 细化后仿真通过率 | 最终成功率 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for name,d in totals.items():
        text+=f"| {name} | {d['initial']} | {d['refined']} | {d['successful']} | {d['refinement_percent']:.2f}% | {d['evaluation_percent']:.2f}% | {d['end_to_end_percent']:.2f}% |\n"
    text+="\nShadow 是 22 关节、18 执行器的右手，使用耦合肌腱与 ±3 Nm 的执行器限制；O30 是 20 关节右手，使用 ±1 Nm 限制。当前 Sharpa 左手配置未设置执行器或关节力矩上限。因此这里比较的是模型、控制与标注整体表现，不能只归因于标注质量。\n\n第 33 类的 O30 和 Shadow 要求四个指节接触，Sharpa 按两个手指接触组检查，要求不完全相同。\n\n"
    text+="| 类型 | Shadow 成功率 | Sharpa 成功率 | O30 基线成功率 |\n| --- | ---: | ---: | ---: |\n"
    for row in rows:
        text+=f"| {row['template']} | {row['Shadow_success_percent']:.2f}% | {row['Sharpa_success_percent']:.2f}% | {row['O30_baseline_success_percent']:.2f}% |\n"
    (args.output/"comparison.md").write_text(text)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
