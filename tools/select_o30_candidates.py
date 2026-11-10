"""仅按调参集选择 O30 候选，冻结选择后再做独立物体验证。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

from tools.optimize_o30_annotations import TARGETS, ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=ROOT/"output/o30_annotation_tuning")
    parser.add_argument("--baseline-run",type=Path,default=ROOT/"output/o30_taxonomy_verified")
    args=parser.parse_args();directory=args.directory
    status=json.loads((directory/"train/status.json").read_text())
    assert status["state"]=="finished" and status["reason"]=="all_objects_processed"
    train_job=json.loads((directory/"train/job.json").read_text())
    baseline_job=json.loads((args.baseline_run/"job.json").read_text())
    assert train_job["objects"]==baseline_job["objects"], "调参集与基线物体不一致"
    for key in ["hand","points","scales"]:
        assert train_job["settings"][key]==baseline_job["settings"][key], key
    if "model_sha256" in train_job and "model_sha256" in baseline_job:
        assert train_job["model_sha256"]==baseline_job["model_sha256"]
    baseline={r["template"]:r for r in csv.DictReader((args.baseline_run/"summary.csv").open())}
    origin={r["candidate"]:r for r in json.loads((directory/"candidate_provenance.json").read_text())}
    candidates=list(csv.DictReader((directory/"train/summary.csv").open()))
    output=directory/"selected_annotations";output.mkdir(exist_ok=True)
    assert not list(output.glob("*.yaml")),"选择已经冻结，不能覆盖"
    records=[]
    for name in TARGETS:
        options=[row for row in candidates if row["template"].startswith(name+"__")]
        best=sorted(options,key=lambda row:(-int(row["successful"]),-int(row["grasps"]),row["template"]))[0]
        assert int(best["initializations"])==int(baseline[name]["initializations"])
        baseline_file=args.baseline_run/"raw_anno_snapshot"/(name+".yaml")
        assert hashlib.sha256(baseline_file.read_bytes()).hexdigest()==origin[best["template"]]["baseline_sha256"]
        improved=int(best["successful"])>int(baseline[name]["successful"])
        source=(directory/"train/raw_anno_snapshot"/(best["template"]+".yaml") if improved
                else baseline_file)
        if improved:
            assert hashlib.sha256(source.read_bytes()).hexdigest()==origin[best["template"]]["sha256"]
        shutil.copy2(source,output/(name+".yaml"))
        records.append({"template":name,"candidate":best["template"] if improved else "baseline",
                        "train_initializations":int(best["initializations"]),
                        "train_baseline_success":int(baseline[name]["successful"]),
                        "train_selected_success":int(best["successful"]) if improved else int(baseline[name]["successful"]),
                        "selected_sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
    result={"selected_at_unix":time.time(),"selection_uses_heldout_data":False,"selection":records}
    (directory/"selection.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
