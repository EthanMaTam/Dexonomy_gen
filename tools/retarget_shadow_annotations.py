"""Retarget the whole Shadow hand, including fingers without contacts.

Writes a separate candidate library for visual review. Numerical feasibility is
reported separately from anatomical similarity and human taxonomy acceptance.
"""

from __future__ import annotations

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
from tools.closed_loop_hand import connect_errors, connect_jacobian, moving_indices, project_passive
from tools.hand_frames import SHADOW_TO_NATIVE
from tools.sharpa_template_editor import ROOT, _contact_entries

FINGERS = [("ff", "index"), ("mf", "middle"), ("rf", "ring"), ("lf", "pinky")]
SEGMENTS = ("proximal", "middle", "distal")


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def anatomical_basis(length, pad):
    length, pad = np.asarray(length, float), np.asarray(pad, float)
    return np.column_stack([np.cross(pad, length), pad, length])


class ShadowRetargeter:
    def __init__(self, hand):
        self.hand = hand
        self.source_dir = ROOT / "assets/hand/shadow"
        self.target_dir = ROOT / "assets/hand" / hand
        self.source = mujoco.MjModel.from_xml_path(str(self.source_dir / "right.xml"))
        self.target = mujoco.MjModel.from_xml_path(str(self.target_dir / "right.xml"))
        self.sd, self.td = mujoco.MjData(self.source), mujoco.MjData(self.target)
        self.sm, self.tm = local_meshes(self.source, self.sd), local_meshes(self.target, self.td)
        self.keypoints = load_yaml(str(self.source_dir / "keypoint.yaml"))
        for name, value in self.keypoints.items():
            if isinstance(value, str): self.keypoints[name] = self.keypoints[value]
        self.closed = bool(self.target.neq)
        self.indices = moving_indices(self.target) if self.closed else np.arange(self.target.nq)
        self.mapping = {f"rh_{short}{seg}": f"{finger}_{seg}"
                        for short, finger in FINGERS for seg in SEGMENTS}
        self.mapping.update({"rh_palm": "hand_base_link",
            "rh_lfmetacarpal": "pinky_metacarpal" if self.closed else "pinky_metacarpals",
            "rh_thproximal": "thumb_proximal" if self.closed else "thumb_metacarpals_base2",
            "rh_thmiddle": "thumb_middle" if self.closed else "thumb_proximal",
            "rh_thdistal": "thumb_distal"})
        self.bases = {}
        for source_body in self.mapping:
            source_pad = [-1, 0, 0] if source_body == "rh_thmiddle" else [0, -1, 0]
            source_basis = anatomical_basis([0, 0, 1], source_pad)
            if "rh_th" in source_body and not self.closed:
                target_basis = anatomical_basis([0, 1, 0], [0, 0, 1])
            else:
                target_basis = anatomical_basis([0, 0, 1], [1, 0, 0])
            self.bases[source_body] = source_basis, target_basis
        self.target.geom_margin[self.target.geom_contype != 0] = .0005
        self.pairs = {(a, b): i for i, (a, b) in enumerate(
            (a, b) for a in range(self.target.nbody) for b in range(a+1, self.target.nbody))}
        sw = np.linalg.norm(self.sd.xpos[self.source.body('rh_ffproximal').id] -
                            self.sd.xpos[self.source.body('rh_lfproximal').id])
        tw = np.linalg.norm(self.td.xpos[self.target.body('index_proximal').id] -
                            self.td.xpos[self.target.body('pinky_proximal').id])
        self.scale = tw / sw

    def source_contacts(self, annotation):
        contacts = []
        for name, raw in annotation["contact"].items():
            bid = self.source.body(name).id
            r = self.sd.xmat[bid].reshape(3, 3)
            for value in _contact_entries(raw):
                if isinstance(value, list):
                    p, n = np.asarray(value, float).reshape(2, 3)
                    p, n = (p-self.sd.xpos[bid]) @ r, n @ r
                else: p, n = np.asarray(self.keypoints[name][int(value)]).reshape(2, 3)
                contacts.append((name, p, n / np.linalg.norm(n)))
        return contacts

    def transfer_contact(self, name, point, normal):
        target_name = self.mapping[name]
        sb, tb = self.bases[name]
        sv, tv = self.sm[name].vertices @ sb, self.tm[target_name].vertices @ tb
        lo, hi = sv.min(0), sv.max(0)
        fraction = np.clip((point @ sb-lo)/np.maximum(hi-lo, 1e-6), 0, 1)
        guess = (tv.min(0) + fraction*(tv.max(0)-tv.min(0))) @ tb.T
        expected = tb @ sb.T @ normal
        mesh = self.tm[target_name]
        points, _, faces = mesh.nearest.on_surface([guess])
        point, face = points[0], faces[0]
        if mesh.face_normals[face] @ expected < .5:
            eligible = np.flatnonzero(mesh.face_normals @ expected > .6)
            if not len(eligible): raise ValueError(f"No matching contact surface: {name}")
            # Restrict to the correct surface side before choosing the nearest
            # location. A bounding-box projection alone can select a knuckle.
            import trimesh
            candidate = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[eligible], process=False)
            points, _, faces = candidate.nearest.on_surface([guess])
            point, face = points[0], eligible[faces[0]]
        return target_name, point, mesh.face_normals[face].copy()

    def initial_pose(self):
        q = np.zeros(self.target.nq)
        for short, finger in FINGERS:
            for old, suffix in [(4, "metacarpal" if self.closed else "mcp_roll"),
                                (3, "proximal" if self.closed else "mcp_pitch"),
                                (2, "middle" if self.closed else "pip"),
                                (1, "distal" if self.closed else "dip")]:
                name = f"{finger}_{suffix}" + ("_joint" if self.closed else "")
                q[self.target.joint(name).qposadr[0]] = self.sd.qpos[self.source.joint(f"rh_{short.upper()}J{old}").qposadr[0]]
        if self.closed:
            for name, value in [("thumb_metacarpal_joint", .35), ("thumb_proximal_joint", .6), ("thumb_distal_joint", .5)]:
                q[self.target.joint(name).qposadr[0]] = value
        else: q[:4] = [.3, 1., .5, .5]
        q = np.clip(q, self.target.jnt_range[:, 0]+1e-8, self.target.jnt_range[:, 1]-1e-8)
        if self.closed:
            try: q = project_passive(self.target, q)
            except ValueError: q[:] = 0
        return q

    def prepare(self, annotation):
        self.sd.qpos[:] = annotation["qpos"]
        mujoco.mj_forward(self.source, self.sd)
        contacts = []
        for name, p, n in self.source_contacts(annotation):
            target_name, tp, tn = self.transfer_contact(name, p, n)
            sid, tid = self.source.body(name).id, self.target.body(target_name).id
            sr = self.sd.xmat[sid].reshape(3, 3)
            contacts.append((target_name, tid, tp, tn,
                             SHADOW_TO_NATIVE @ (sr @ p+self.sd.xpos[sid]) * self.scale,
                             SHADOW_TO_NATIVE @ sr @ n))
        bones = []
        active_bodies = {self.mapping[name] for name in annotation.get('contact', {})}
        for name, target_name in self.mapping.items():
            if name in {"rh_palm", "rh_lfmetacarpal"}: continue
            sid, tid = self.source.body(name).id, self.target.body(target_name).id
            sb, tb = self.bases[name]
            desired = SHADOW_TO_NATIVE @ self.sd.xmat[sid].reshape(3, 3) @ sb
            # Full-hand posture is a requirement even for fingers with no contacts.
            weight = .85 if target_name in active_bodies else 1.2
            bones.append((tid, tb, desired, weight))
        return contacts, bones

    def fit(self, annotation, max_evaluations=250):
        contacts, bones = self.prepare(annotation)
        initial = self.initial_pose()
        indices, model, data = self.indices, self.target, self.td
        n = len(contacts)
        contact_pairs = [(a,b) for a in range(n) for b in range(a+1,n)
                         if contacts[a][1] != contacts[b][1]]
        pair_weight = 65 / np.sqrt(max(len(contact_pairs), 1))

        def terms(value, jac=False):
            data.qpos[:] = initial
            data.qpos[indices] = value
            mujoco.mj_forward(model, data)
            residuals, jacobians, positions, position_jacs = [], [], [], []
            for tid, basis, desired, weight in bones:
                r = data.xmat[tid].reshape(3, 3)
                for axis, w in [(2,weight), (1,.35*weight)]:
                    direction = r @ basis[:, axis]
                    residuals.append(w*(direction-desired[:,axis]))
                    if jac:
                        jr = np.zeros((3, model.nv))
                        mujoco.mj_jacBody(model, data, None, jr, tid)
                        jacobians.append(-w*skew(direction) @ jr)
            for _, tid, p, normal, _, goal_normal in contacts:
                r = data.xmat[tid].reshape(3, 3)
                point, normal = r @ p+data.xpos[tid], r @ normal
                positions.append(point)
                residuals.append(.5*(normal-goal_normal)/np.sqrt(n))
                if jac:
                    jp, jr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
                    mujoco.mj_jac(model, data, jp, jr, point, tid)
                    position_jacs.append(jp)
                    jacobians.append(-.5/np.sqrt(n)*skew(normal) @ jr)
            for a,b in contact_pairs:
                residuals.append(pair_weight*(positions[a]-positions[b]-contacts[a][4]+contacts[b][4]))
                if jac: jacobians.append(pair_weight*(position_jacs[a]-position_jacs[b]))
            if self.closed:
                residuals.append(200000*connect_errors(model,data).ravel())
                if jac: jacobians.append(200000*connect_jacobian(model,data))
            collision = np.zeros(len(self.pairs))
            cj = np.zeros((len(self.pairs),model.nv))
            for c in data.contact:
                a,b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if a == b: continue
                if not self.closed and (model.body_parentid[a] == b or model.body_parentid[b] == a):
                    continue  # O30 mounting housings overlap in the vendor CAD.
                pair = self.pairs[tuple(sorted((a,b)))]
                depth = .0002-c.dist
                if depth <= collision[pair]: continue
                collision[pair] = depth
                if jac:
                    ja,jb = np.zeros((3,model.nv)), np.zeros((3,model.nv))
                    mujoco.mj_jac(model,data,ja,None,c.pos,a)
                    mujoco.mj_jac(model,data,jb,None,c.pos,b)
                    cj[pair] = -c.frame[:3] @ (jb-ja)
            residuals.extend([2000*collision, .015*(value-initial[indices])])
            if jac:
                jacobians.extend([2000*cj, .015*np.eye(model.nv)[indices]])
                return np.concatenate(jacobians)[:,indices]
            return np.concatenate(residuals)

        fitted=least_squares(terms, initial[indices], jac=lambda v:terms(v,True),
                            bounds=(model.jnt_range[indices,0],model.jnt_range[indices,1]),
                            max_nfev=max_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        terms(fitted.x)
        q=data.qpos.copy()
        if self.closed: q=project_passive(model,q)
        q=np.round(q,9)
        data.qpos[:]=q
        mujoco.mj_forward(model,data)
        grouped={}
        for name,tid,p,normal,_,_ in contacts:
            r=data.xmat[tid].reshape(3,3)
            grouped.setdefault(name,[]).append(np.round(np.r_[r@p+data.xpos[tid],r@normal],9).tolist())
        result={'qpos':q.tolist(),'contact':{k:v[0] if len(v)==1 else v for k,v in grouped.items()}}
        if 'required_cbody' in annotation:
            result['required_cbody']=[[self.mapping[name] for name in group] for group in annotation['required_cbody']]
        return result, self.assess(annotation,result), fitted.nfev

    def assess(self, reference, candidate):
        contacts,bones=self.prepare(reference)
        self.td.qpos[:]=candidate['qpos']
        mujoco.mj_forward(self.target,self.td)
        errors={}
        for tid,basis,desired,_ in bones:
            actual=self.td.xmat[tid].reshape(3,3) @ basis[:,2]
            errors[self.target.body(tid).name]=float(np.degrees(np.arccos(np.clip(actual@desired[:,2],-1,1))))
        loop=float(np.linalg.norm(connect_errors(self.target,self.td),axis=1).max()) if self.closed else 0.
        by_body={}
        for name,raw in candidate['contact'].items():
            by_body[name]=[np.asarray(v,float) for v in _contact_entries(raw)]
        matched=[]; normal_errors=[]
        used={}
        for name,_,_,_,goal,normal in contacts:
            index=used.get(name,0)
            if index>=len(by_body.get(name,[])): continue
            value=by_body[name][index];used[name]=index+1
            matched.append((value[:3],goal))
            normal_errors.append(float(np.degrees(np.arccos(np.clip(value[3:]@normal,-1,1)))))
        relative_errors=[np.linalg.norm((a[0]-b[0])-(a[1]-b[1]))
                         for i,a in enumerate(matched) for b in matched[i+1:]]
        depth=max([max(0.,-float(c.dist)) for c in self.td.contact
                   if self.closed or not (
                       self.target.body_parentid[self.target.geom_bodyid[c.geom1]] == self.target.geom_bodyid[c.geom2] or
                       self.target.body_parentid[self.target.geom_bodyid[c.geom2]] == self.target.geom_bodyid[c.geom1])],default=0.)
        return {'mean_bone_direction_error_deg':float(np.mean(list(errors.values()))),
                'max_bone_direction_error_deg':max(errors.values()),'bone_direction_errors_deg':errors,
                'mean_contact_normal_error_deg':float(np.mean(normal_errors)),
                'contact_layout_rms_error_m':float(np.sqrt(np.mean(np.square(relative_errors)))),
                'reference_contacts':len(contacts),'matched_contacts':len(matched),
                'max_connect_error_m':loop,'max_self_penetration_m':depth,
                'human_taxonomy_review_complete':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand',required=True,choices=['linker_o30_right','zte_hand'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--templates',nargs='+')
    parser.add_argument('--max-evaluations',type=int,default=250)
    parser.add_argument('--skip-existing',action='store_true',help='Preserve existing annotations and complete missing numbered templates')
    args=parser.parse_args(argv)
    retargeter=ShadowRetargeter(args.hand)
    args.output.mkdir(parents=True,exist_ok=True)
    yaml=YAML(); yaml.default_flow_style=None
    paths=sorted(retargeter.source_dir.joinpath('raw_anno').glob('*.yaml'))
    paths=[p for p in paths if p.stem.split('_')[0].isdigit() and int(p.stem.split('_')[0])<=33]
    if args.templates:
        paths=[p for p in paths if p.stem in args.templates]
        if {p.stem for p in paths}!=set(args.templates): parser.error('Unknown Shadow template')
    report_path=args.output.parent/'retarget_report.json'
    reports=json.loads(report_path.read_text())['records'] if args.skip_existing and report_path.exists() else []
    for path in paths:
        output=args.output/path.name
        if output.exists():
            if args.skip_existing:
                print(path.stem,'preserved',flush=True)
                continue
            parser.error(f'Candidate already exists: {output}; use a new output directory')
        annotation=load_yaml(str(path))
        result,assessment,nfev=retargeter.fit(annotation,args.max_evaluations)
        previous_path=retargeter.target_dir/'raw_anno'/path.name
        old=retargeter.assess(annotation,load_yaml(str(previous_path))) if previous_path.exists() else None
        with output.open('x') as stream: yaml.dump(result,stream)
        reports=[r for r in reports if r['template'] != path.stem]
        reports.append({'template':path.stem,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'candidate_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                        'before':old,'after':assessment,'solver_evaluations':nfev})
        report_path.write_text(json.dumps({
            'source_hand':'shadow','target_hand':args.hand,'display_native_to_shadow':SHADOW_TO_NATIVE.T.tolist(),
            'source_model_sha256':hashlib.sha256((retargeter.source_dir/'right.xml').read_bytes()).hexdigest(),
            'target_model_sha256':hashlib.sha256((retargeter.target_dir/'right.xml').read_bytes()).hexdigest(),
            'method':'whole-hand bone directions, pad orientation, contact-relative geometry, collision and loop constraints',
            'records':reports},indent=2)+'\n')
        print(path.stem, 'bone error',round(old['mean_bone_direction_error_deg'],1) if old else 'no baseline','->',
              round(assessment['mean_bone_direction_error_deg'],1), 'deg',flush=True)


if __name__=='__main__': main()
