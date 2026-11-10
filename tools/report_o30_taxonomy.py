"""汇总 O30 全抓取分类库的几何检查、接触数量和逐类仿真结果。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from dexonomy.util.file_util import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "output/o30_taxonomy_verified")
    args = parser.parse_args()
    directory = ROOT / "assets/hand/linker_o30_right"
    model_hash = hashlib.sha256((directory / "right.xml").read_bytes()).hexdigest()
    job = json.loads((args.run / "job.json").read_text())
    if "model_sha256" in job:
        assert model_hash == job["model_sha256"], "Hand model changed since the run"
    status = json.loads((args.run / "status.json").read_text())
    assert status["state"] == "finished" and status["reason"] == "all_objects_processed", status
    summary = {row["template"]: row for row in csv.DictReader((args.run / "summary.csv").open())}
    geometry = {row["name"]: row for row in json.loads((args.run / "asset_checks.json").read_text())["templates"]}
    provenance = {row["template"]: row for row in json.loads((directory / "taxonomy_provenance.json").read_text())}
    paths = sorted((directory / "raw_anno").glob("*.yaml"), key=lambda p:
                   (int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 1000, p.stem))
    ids = {int(p.stem.split("_")[0]) for p in paths if p.stem.split("_")[0].isdigit()}
    assert ids == set(range(1, 34)), ids
    rows = []
    for path in paths:
        assert path.read_bytes() == (args.run / "raw_anno_snapshot" / path.name).read_bytes(), path
        annotation = load_yaml(str(path))
        counts = summary[path.stem]
        successful = int(counts["successful"])
        row = {
            "template": path.stem, "contact_points": geometry[path.stem]["contacts"],
            "contact_bodies": ";".join(annotation["contact"]),
            "geometry": "通过", "initializations": int(counts["initializations"]),
            "refined": int(counts["grasps"]), "successful": successful,
            "simulation": "有成功样本" if successful else "本次无成功样本，需继续改进",
            "origin": ("保留原始模板" if path.stem not in provenance else
                       "优化并通过独立物体验证，分类语义待人工确认" if provenance[path.stem]["status"] == "优化并通过独立物体验证" else
                       "保留原始模板" if provenance[path.stem]["status"] == "保留已有标注" else
                       "自动迁移，分类语义待人工确认"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        rows.append(row)
    with (directory / "taxonomy_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["successful"] > 0 for row in rows if row["template"].split("_")[0].isdigit())
    totals = {key: sum(row[key] for row in rows) for key in ["initializations", "refined", "successful"]}
    note = ("# O30 右手全分类标注验证\n\n"
            "覆盖 33 类编号抓取，另保留 fingertip_large，共 34 份标注。\n\n"
            "几何检查通过只说明接触点、法向和关节数据合格；物理仿真成功也不等同于"
            "抓取分类语义已经人工确认。新迁移的 31 类仍应结合总览图审阅。\n\n"
            f"编号分类中 {passed}/33 类在本次小规模测试中得到成功样本。"
            f"合计 {totals['initializations']} 个候选、{totals['refined']} 个有效抓取、"
            f"{totals['successful']} 个物理仿真成功样本。\n\n"
            "本次测试使用确定性选择的 6 个 DGN 物体、每个 3 个尺度；没有成功样本"
            "不证明该类型不可实现，但该类型不能标记为本次已验证可用。\n\n"
            "| 模板 | 接触点数 | 初始候选 | 有效抓取 | 仿真成功 | 状态 |\n"
            "| --- | ---: | ---: | ---: | ---: | --- |\n")
    for row in rows:
        note += f"| {row['template']} | {row['contact_points']} | {row['initializations']} | {row['refined']} | {row['successful']} | {row['simulation']} |\n"
    (directory / "taxonomy_validation.md").write_text(note)
    (args.run / "verification.json").write_text(json.dumps(
        {"numbered_categories": 33, "numbered_categories_with_success": passed,
         "templates": len(rows), "totals": totals,
         "model_sha256": model_hash,
         "failed_categories": [r["template"] for r in rows if not r["successful"]],
         "annotations_match_snapshot": True}, ensure_ascii=False, indent=2)+"\n")
    print(note)


if __name__ == "__main__":
    main()
