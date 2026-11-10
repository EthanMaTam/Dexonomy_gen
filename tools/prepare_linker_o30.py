"""Build a reproducible Dexonomy simulation model from the official O30 right URDF."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco

from tools.linker_o30_preview import ROOT, MODEL_RELATIVE, prepare_model


def main():
    destination = ROOT / "assets/hand/linker_o30_right"
    destination.mkdir(parents=True, exist_ok=True)
    preview, revision = prepare_model(ROOT / "output/linker_o30_conversion")
    tree = ET.parse(preview)
    root = tree.getroot()
    ET.SubElement(root, "option", cone="elliptic", impratio="10")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", damping="0.05", armature="0.0002", frictionloss="0.01")
    actuators = ET.SubElement(root, "actuator")
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if not name:
            continue
        # The source ring MCP roll has zero effort. Use the same 1 Nm simulation
        # limit as the other source joints; this is not hardware calibration.
        joint.set("actuatorfrcrange", "-1 1")
        ET.SubElement(
            actuators, "position", name=name + "_position", joint=name,
            kp="5", forcerange="-1 1", ctrlrange=joint.attrib["range"],
        )
    # The source uses complete CAD shells as collision meshes. Their convex
    # hulls overlap at the mounted joints, even in the zero pose. Exclude only
    # these structural neighbours, as the existing Shadow model also does.
    contact = ET.SubElement(root, "contact")
    structural_pairs = [("hand_base_link", "thumb_metacarpals_base2")] + [
        ("hand_base_link", finger + "_proximal")
        for finger in ("index", "middle", "ring", "pinky")
    ] + [(a + "_metacarpals", b + "_metacarpals")
         for a, b in (("index", "middle"), ("middle", "ring"), ("ring", "pinky"))]
    for first, second in structural_pairs:
        ET.SubElement(contact, "exclude", body1=first, body2=second)
    # Keep source meshes external and paths relative to this generated model.
    meshdir = Path(root.find("compiler").attrib["meshdir"])
    import os
    root.find("compiler").set("meshdir", os.path.relpath(meshdir, destination))
    ET.indent(tree, space="  ")
    tree.write(destination / "right.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(destination / "right.xml"))
    assert model.nq == model.nu == 20
    metadata = {
        "repository": "https://github.com/linker-bot/linkerhand-urdf",
        "revision": revision, "source": str(MODEL_RELATIVE),
        "hand": "right", "moving_joints": 20,
        "changes": ["Preserve visual and collision meshes; assign unique geometry names",
                    "20 position actuators: kp=5, force limit +/-1 Nm",
                    "ring_mcp_roll: replace source zero effort with simulation assumption +/-1 Nm",
                    "joint damping=0.05, armature=0.0002, frictionloss=0.01"],
        "hardware_calibrated": False,
        "structural_collision_exclusions": structural_pairs,
    }
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {destination / 'right.xml'}: nq={model.nq}, nu={model.nu}")


if __name__ == "__main__":
    main()
