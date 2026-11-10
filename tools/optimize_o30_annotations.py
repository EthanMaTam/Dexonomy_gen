"""生成 O30 弱项标注的独立候选，保持接触指节和全局评估阈值不变。"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from ruamel.yaml import YAML
from scipy.optimize import least_squares

from dexonomy.util.file_util import load_yaml
from tools.annotate_o30_taxonomy import local_meshes
from tools.sharpa_template_editor import ROOT


TARGETS = ["14_Tripod", "24_Tip_Pinch", "27_Quadpod", "30_Palmar"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/o30_annotation_tuning/candidates")
    parser.add_argument("--annotation-dir", type=Path, help="Use archived baseline annotations for reproduction")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.yaml")):
        parser.error("候选目录非空，请使用新目录，避免覆盖已有标注")
    directory = ROOT / "assets/hand/linker_o30_right"
    annotation_dir = args.annotation_dir if args.annotation_dir else directory / "raw_anno"
    model = mujoco.MjModel.from_xml_path(str(directory / "right.xml"))
    data = mujoco.MjData(model)
    meshes = local_meshes(model, data)
    model.geom_margin[model.geom_contype != 0] = .001
    pairs = {(a,b): i for i,(a,b) in enumerate(
        ((a,b) for a in range(model.nbody) for b in range(a+1,model.nbody)))}
    yaml = YAML(); yaml.default_flow_style = None
    provenance = []

    def store(name, payload, record):
        path = args.output / (name+".yaml")
        with path.open("x") as stream:
            yaml.dump(payload,stream)
        provenance.append({"candidate":name, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), **record})
        print(name,flush=True)

    for name in TARGETS:
        source = annotation_dir / (name+".yaml")
        base = load_yaml(str(source))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        baseline = np.array(base["qpos"])
        pads = {}
        for body in base["contact"]:
            mesh = meshes[body]
            if body == "hand_base_link":
                center, direction = np.array([0.,0.,.055]), np.array([1.,0.,0.])
            elif body.startswith("thumb_"):
                center = np.array([0., .015 if body.endswith("distal") else .025, 0.])
                direction = np.array([0.,0.,1.])
            else:
                center = np.array([0.,0.,.015 if body.endswith("distal") else .022])
                direction = np.array([1.,0.,0.])
            hits,_,faces = mesh.ray.intersects_location([center+.1*direction],[-direction],multiple_hits=False)
            assert len(hits)==1,(name,body)
            pads[body]=(hits[0],mesh.face_normals[faces[0]])

        def payload(q):
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            contacts={}
            for body,(point,normal) in pads.items():
                bid=model.body(body).id;rotation=data.xmat[bid].reshape(3,3)
                contacts[body]=np.round(np.r_[rotation@point+data.xpos[bid],rotation@normal],7).tolist()
            result={"qpos":np.round(q,7).tolist(),"contact":contacts}
            if "required_cbody" in base:result["required_cbody"]=base["required_cbody"]
            return result

        record={"template":name,"baseline_sha256":source_hash,
                "contact_bodies_preserved":True,"model_or_threshold_changes":False}
        store(name+"__pad_center",payload(baseline),{**record,"method":"保持原姿态，仅将接触点投影至指腹中心及掌心"})
        active={body.split("_")[0] for body in pads if body!="hand_base_link"}
        indices=[i for i in range(model.njnt) if model.joint(i).name.split("_")[0] in active]
        for radius in [.025,.04,.055]:
            center0=np.array([.065,.01,.11])
            def residual(value):
                q=baseline.copy();q[indices]=value[:-3]
                data.qpos[:]=q;mujoco.mj_forward(model,data)
                centers=[]
                for body,(point,normal) in pads.items():
                    bid=model.body(body).id;r=data.xmat[bid].reshape(3,3)
                    weight=2 if body=="hand_base_link" else 1
                    centers.extend(100*weight*(r@(point+radius*normal)+data.xpos[bid]-value[-3:]))
                collision=np.zeros(len(pairs))
                for c in data.contact:
                    a,b=sorted((int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])))
                    if model.body_parentid[b]==a or model.body_parentid[a]==b:continue
                    collision[pairs[a,b]]=max(collision[pairs[a,b]],.0003-c.dist)
                return np.r_[centers,.015*(value[:-3]-baseline[indices]),.2*(value[-3:]-center0),1500*collision]
            low=np.r_[model.jnt_range[indices,0]+1e-5,[.03,-.025,.045]]
            high=np.r_[model.jnt_range[indices,1]-1e-5,[.14,.06,.18]]
            initial=np.clip(np.r_[baseline[indices],center0],low+1e-8,high-1e-8)
            fitted=least_squares(residual,initial,bounds=(low,high),max_nfev=600,
                                 ftol=1e-9,xtol=1e-9,gtol=1e-9)
            q=baseline.copy();q[indices]=fitted.x[:-3]
            store(name+f"__sphere_{int(radius*1000):03}",payload(q),
                  {**record,"method":"保持接触指节与数量，优化对向及张开尺度",
                   "radius_m":radius,"fit_cost":float(fitted.cost)})
        if name=="14_Tripod":
            wider=load_yaml(str(directory/"raw_anno/fingertip_large.yaml"))
            assert list(wider["contact"])==list(base["contact"])
            store(name+"__wide_seed",wider,{**record,"method":"复用同样三指接触拓扑的较宽初始姿态"})
    (args.output.parent/"candidate_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
