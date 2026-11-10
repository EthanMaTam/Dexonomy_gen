"""Preview the official O30 right hand using MuJoCo forward kinematics and Viser."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import viser

from tools.sharpa_template_editor import HandView


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = ROOT / "third_party/linkerhand-urdf"
MODEL_RELATIVE = Path(
    "O30/urdf_0803-right/src/linkerhand_O30i_right.urdf/"
    "linkerhand_O30i_right-0803.urdf"
)
EXPECTED_REVISION = "49f26021b0a91ff5c33150df22970c07bb548672"


def prepare_model(destination: Path) -> tuple[Path, str]:
    source = MODEL_REPO / MODEL_RELATIVE
    if not (MODEL_REPO / ".git").exists() or not source.is_file():
        raise FileNotFoundError(
            "O30 model submodule is missing or incomplete. Run from the project root: "
            "git submodule update --init third_party/linkerhand-urdf"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise ValueError(f"Expected model revision {EXPECTED_REVISION}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("O30 source repository has local changes; restore the pinned assets before conversion")
    robot = ET.parse(source)
    root = robot.getroot()
    joints = [joint for joint in root.findall("joint") if joint.get("type") != "fixed"]
    if root.get("name") != "linkerhand_O30i_right.urdf" or len(joints) != 20:
        raise ValueError("Expected the official O30i right hand with 20 moving joints")
    # Preserve both visual and collision geometry, and resolve the vendor mesh paths.
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension, "compiler", discardvisual="false", fusestatic="false",
        meshdir=str(source.parent / "meshes"),
    )
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for index, geom in enumerate(link.findall(kind)):
                geom.set("name", f"{link.attrib['name']}_{kind}_{index}")
    for mesh in root.findall(".//mesh"):
        path = (source.parent / mesh.attrib["filename"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path))
    destination.mkdir(parents=True, exist_ok=True)
    converted = destination / "right.urdf"
    robot.write(converted, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(converted))
    if model.nq != 20:
        raise ValueError(f"Expected 20 qpos values, found {model.nq}")
    xml_path = destination / "right.xml"
    mujoco.mj_saveLastXML(str(xml_path), model)
    return xml_path, revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8047)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    xml_path, revision = prepare_model(ROOT / "output/linker_o30_preview")
    if args.check:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print(f"OK: O30 right, nq={model.nq}, meshes={model.nmesh}, revision={revision}")
        return
    server = viser.ViserServer(host=args.host, port=args.port, label="Linker Hand O30 right")
    hand = HandView(server, "o30_right", xml_path, (0.0, 0.0, 0.0))
    server.gui.add_markdown(
        "## Linker Hand O30 — Right\n"
        "Official asset name: **O30i right / 0803**.\n\n"
        "MuJoCo forward-kinematics preview; sliders are joint angles in radians. "
        "For the adapted actuated model and validated grasp templates, see "
        "assets/hand/linker_o30_right/README.md.\n\n"
        "Drag to rotate; scroll to zoom.\n\n"
        f"Source revision: `{revision[:12]}`"
    )
    sliders = []
    lock = threading.RLock()

    def update_joint(index: int, value: float) -> None:
        with lock:
            qpos = hand.data.qpos.copy()
            qpos[index] = value
            hand.set_qpos(qpos)

    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        with server.gui.add_folder(finger.title()):
            for joint_id in range(hand.model.njnt):
                name = hand.model.joint(joint_id).name
                if not name.startswith(finger + "_"):
                    continue
                index = int(hand.model.jnt_qposadr[joint_id])
                low, high = hand.model.jnt_range[joint_id]
                slider = server.gui.add_slider(
                    name, min=float(low), max=float(high), step=0.001,
                    initial_value=float(hand.data.qpos[index]),
                )
                slider.on_update(lambda event, i=index: update_joint(i, event.target.value))
                sliders.append((slider, index))
    reset = server.gui.add_button("Reset to zero pose")

    @reset.on_click
    def reset_pose(_) -> None:
        with lock:
            for slider, index in sliders:
                slider.value = float(hand.model.qpos0[index])
            hand.set_qpos(hand.model.qpos0.copy())

    collisions = server.gui.add_checkbox("Show collision meshes", initial_value=False)

    @collisions.on_update
    def show_collisions(event) -> None:
        for handle in hand.collision_handles.values():
            handle.visible = event.target.value

    center = hand.data.geom_xpos.mean(axis=0)

    @server.on_client_connect
    def configure_camera(client) -> None:
        client.camera.look_at = center
        client.camera.position = center + np.asarray([0.30, -0.35, 0.22])

    print(f"Open http://{args.host}:{args.port}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
