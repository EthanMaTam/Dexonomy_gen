"""将本地 ZTE V1.0 压缩包的闭环 MJCF 转为独立手模型，保留被动关节与驱动拓扑。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
from ruamel.yaml import YAML

from tools.annotate_o30_taxonomy import local_meshes

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "86453deb218dd653c7c1572cc1a19fd8f67f7d138a88c10e01dc1610289b25a4"
SOURCE = ROOT / "third_party/zte_hand/right_hand_V1.0"
DESTINATION = ROOT / "assets/hand/zte_hand"
FINGERS = {"thumb": "thumb", "IF": "index", "MF": "middle", "RF": "ring", "LF": "pinky"}


def name_map():
    bodies = {"arm_Y_Link": "hand_base_link"}
    joints = {}
    for old, finger in FINGERS.items():
        for suffix, segment in [("4", "metacarpal"), ("3", "proximal"), ("2", "middle"), ("1", "distal"),
                                ("1_1", "tip_pivot1"), ("1_2", "tip_pivot2"),
                                ("3_1", "coupler1"), ("3_2", "coupler2")]:
            bodies[f"{old}_Link{suffix}"] = f"{finger}_{segment}"
            joints[f"{old}_J{suffix}"] = f"{finger}_{segment}_joint"
    return bodies, joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "right_hand_V1.0.zip")
    args = parser.parse_args()
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        parser.error(f"压缩包版本不符：预期 {ARCHIVE_SHA256}，实际 {digest}")
    target = SOURCE.parent.resolve()
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            if not (target / entry.filename).resolve().is_relative_to(target):
                parser.error(f"非法压缩包路径：{entry.filename}")
        archive.extractall(target)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SOURCE / "xml/right_hand_v1.0.xml")
    model_root = tree.getroot()
    original = mujoco.MjModel.from_xml_path(str(SOURCE / "xml/right_hand_v1.0.xml"))
    assert (original.nq, original.nu, original.neq) == (42, 17, 10)
    body_names, joint_names = name_map()
    hand = copy.deepcopy(model_root.find(".//body[@name='arm_Y_Link']"))
    hand.set("name", "hand_base_link")
    hand.set("pos", "0 0 0")
    hand.set("quat", "1 0 0 0")
    hand.remove(hand.find("joint"))
    hand.remove(hand.find("inertial"))  # Fixed root: source aggregate includes the removed wrist.
    palm_origin = np.array([-.007, -.0385, .0242])
    for element in list(hand):
        if element.tag == "geom" and element.get("mesh") == "arm_Y_Link":
            hand.remove(element)
            continue
        if element.tag in ("geom", "body"):
            pos = np.fromstring(element.get("pos", "0 0 0"), sep=" ") - palm_origin
            element.set("pos", " ".join(f"{x:.10g}" for x in pos))
    world = model_root.find("worldbody")
    world.clear()
    world.append(hand)
    for body in model_root.findall(".//body"):
        body.set("name", body_names.get(body.get("name"), body.get("name")))
    for joint in model_root.findall(".//body/joint"):
        joint.set("name", joint_names[joint.get("name")])
    for equality in model_root.findall("equality/*"):
        for key in ("body1", "body2"):
            equality.set(key, body_names[equality.get(key)])
    actuators = model_root.find("actuator")
    for actuator in list(actuators):
        old_joint = actuator.get("joint")
        if old_joint.startswith("arm_"):
            actuators.remove(actuator)
            continue
        actuator.set("joint", joint_names[old_joint])
        actuator.set("name", joint_names[old_joint] + "_position")
        # Same provisional controller scale as O30; vendor URDF effort fields are zero.
        actuator.set("kp", "5")
        actuator.set("kv", "0.1")
        actuator.set("forcerange", "-1 1")
        joint = model_root.find(f".//joint[@name='{joint_names[old_joint]}']")
        actuator.set("ctrlrange", joint.get("range"))
    model_root.find("compiler").set("meshdir", os.path.relpath(SOURCE / "meshes", DESTINATION))
    # Remove global self-collision suppression. Only mechanical neighbours are excluded below.
    contact = model_root.find("contact")
    source_exclusions = len(contact)
    contact.clear()
    structural_pairs = []
    for finger in FINGERS.values():
        # The fixed palm includes the mounted finger housings; their CAD convex
        # hulls overlap the two bearing links even at the source rest pose.
        for segment in ("metacarpal", "proximal"):
            pair = ("hand_base_link", f"{finger}_{segment}")
            ET.SubElement(contact, "exclude", body1=pair[0], body2=pair[1])
            structural_pairs.append(pair)
        # Linkage is internal transmission, not an external contact pad.
        for segment in ("coupler1", "coupler2"):
            body = model_root.find(f".//body[@name='{finger}_{segment}']")
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "1")
    # Give every geom a stable unique name for contact picking.
    for body in model_root.findall(".//body"):
        for i, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{body.get('name')}_geom_{i}")
    ET.indent(tree, space="  ")
    tree.write(DESTINATION / "right.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(DESTINATION / "right.xml"))
    assert (model.nq, model.nu, model.neq) == (40, 15, 10)
    data = mujoco.MjData(model)
    meshes = local_meshes(model, data)
    skeleton = {}
    for name, mesh in meshes.items():
        low, high = mesh.bounds
        center = (low + high) / 2
        axis = int(np.argmax(high-low))
        a, b = center.copy(), center.copy()
        a[axis] = low[axis] + .15*(high[axis]-low[axis])
        b[axis] = high[axis] - .15*(high[axis]-low[axis])
        skeleton[name] = [np.round(np.r_[a, b], 7).tolist()]
    yaml = YAML()
    yaml.default_flow_style = None
    for filename, payload in {
        "skeleton.yaml": skeleton,
        "body_group.yaml": {"body_group": [[n for n in meshes if n.startswith(f+"_")]
                                              for f in FINGERS.values()] + [["hand_base_link"]]},
        "keypoint.yaml": {},
    }.items():
        with (DESTINATION / filename).open("w") as stream:
            yaml.dump(payload, stream)
    metadata = {
        "source_archive": args.archive.name, "archive_sha256": digest,
        "source_xml_sha256": hashlib.sha256((SOURCE / "xml/right_hand_v1.0.xml").read_bytes()).hexdigest(),
        "source_urdf_sha256": hashlib.sha256((SOURCE / "urdf/right_hand_v1.0.urdf").read_bytes()).hexdigest(),
        "source_model": "zte-hand_right", "source_nq": 42, "source_actuators": 17,
        "nq": model.nq, "actuators": model.nu, "connect_equalities": model.neq,
        "body_name_map": body_names, "joint_name_map": joint_names,
        "removed_wrist_joints": ["arm_X_J", "arm_Y_J"],
        "root_shift_in_source_arm_Y_frame_m": palm_origin.tolist(),
        "source_collision_exclusions_removed": source_exclusions,
        "structural_collision_exclusions": structural_pairs,
        "collision_changes": "内部传动连杆仅显示，手指外壳及手掌保留碰撞；恢复不同手指间自碰撞",
        "controller_assumptions": {"kp": 5, "kv": .1, "force_limit_Nm": 1},
        "hardware_calibrated": False,
        "license": "用户提供的压缩包未包含许可证；未推断公开下载来源或再分发许可",
        "mesh_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SOURCE/"meshes").glob("*.STL"))},
    }
    (DESTINATION / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    print(f"ZTE hand: nq={model.nq}, nu={model.nu}, neq={model.neq}; original self-collision exclusions removed={source_exclusions}")
    mujoco.mj_forward(model, data)
    for c in data.contact:
        if c.dist < -1e-5:
            print("rest collision", model.body(model.geom_bodyid[c.geom1]).name,
                  model.body(model.geom_bodyid[c.geom2]).name, float(c.dist))


if __name__ == "__main__":
    main()
