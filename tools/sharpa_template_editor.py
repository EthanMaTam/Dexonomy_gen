#!/usr/bin/env python3
"""Side-by-side Shadow reference and Sharpa template editor."""

from __future__ import annotations

import argparse
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from ruamel.yaml import YAML
from transforms3d.quaternions import mat2quat

from dexonomy.util.file_util import load_yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHADOW_XML = ROOT / "assets/hand/shadow/right.xml"
DEFAULT_SHADOW_TEMPLATES = ROOT / "assets/hand/shadow/raw_anno"
DEFAULT_SHADOW_KEYPOINTS = ROOT / "assets/hand/shadow/keypoint.yaml"
DEFAULT_SHARPA_XML = (
    ROOT
    / "third_party/sharpa-urdf-usd-xml/wave_01/left_sharpa_wave"
    / "left_sharpa_wave.xml"
)
DEFAULT_OUTPUT = ROOT / "assets/hand/sharpa_left/raw_anno"
SHARPA_DISPLAY_ROTATION = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)


def _contact_entries(raw_contact: object) -> list[object]:
    if not isinstance(raw_contact, list):
        return [raw_contact]
    if len(raw_contact) == 6 and any(isinstance(value, float) for value in raw_contact):
        return [raw_contact]
    return raw_contact


def _geom_mesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.mesh(int(model.geom_dataid[geom_id]))
        vertices = model.mesh_vert[
            mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]
        ]
        faces = model.mesh_face[
            mesh.faceadr[0] : mesh.faceadr[0] + mesh.facenum[0]
        ]
        return trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    raise ValueError(f"Unsupported geom type {geom_type} for geom {geom_id}")


class HandView:
    def __init__(
        self,
        server: viser.ViserServer,
        name: str,
        xml_path: Path,
        offset: tuple[float, float, float],
        display_rotation: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.display_rotation = (
            np.eye(3) if display_rotation is None else display_rotation
        )
        self.root = server.scene.add_frame(
            f"/{name}",
            show_axes=True,
            axes_length=0.035,
            axes_radius=0.0015,
            position=offset,
            wxyz=mat2quat(self.display_rotation),
        )
        self.visual_handles: dict[int, object] = {}
        self.collision_handles: dict[int, object] = {}
        self.collision_meshes: dict[int, trimesh.Trimesh] = {}

        mujoco.mj_forward(self.model, self.data)
        for geom_id in range(self.model.ngeom):
            mesh = _geom_mesh(self.model, geom_id)
            if self.model.geom_contype[geom_id] != 0:
                self.collision_meshes[geom_id] = mesh
                geom_name = self.model.geom(geom_id).name or f"geom_{geom_id}"
                self.collision_handles[geom_id] = server.scene.add_mesh_simple(
                    f"/{name}/collision/{geom_id}_{geom_name}",
                    vertices=np.asarray(mesh.vertices, dtype=np.float32),
                    faces=np.asarray(mesh.faces, dtype=np.uint32),
                    color=(255, 165, 0),
                    wireframe=True,
                    opacity=0.35,
                    side="double",
                    visible=False,
                )
                continue
            rgba = self.model.geom_rgba[geom_id]
            color = tuple(int(255 * channel) for channel in rgba[:3])
            geom_name = self.model.geom(geom_id).name or f"geom_{geom_id}"
            self.visual_handles[geom_id] = server.scene.add_mesh_simple(
                f"/{name}/visual/{geom_id}_{geom_name}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
                side="double",
            )
        self.update_visuals()

    def set_qpos(self, qpos: np.ndarray) -> None:
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"{self.name}: expected {self.model.nq} qpos values, got {qpos.shape}")
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.update_visuals()

    def update_visuals(self) -> None:
        for geom_id, handle in {**self.visual_handles, **self.collision_handles}.items():
            handle.position = self.data.geom_xpos[geom_id]
            handle.wxyz = mat2quat(self.data.geom_xmat[geom_id].reshape(3, 3))

    def body_pose(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.xpos[body_id].copy(),
            self.data.xmat[body_id].reshape(3, 3).copy(),
        )

    def raycast(
        self, ray_origin_world: np.ndarray, ray_direction: np.ndarray, elastomer_only: bool
    ) -> tuple[int, int, np.ndarray, np.ndarray] | None:
        ray_origin = (
            ray_origin_world - np.asarray(self.root.position)
        ) @ self.display_rotation
        direction = ray_direction / np.linalg.norm(ray_direction)
        direction = direction @ self.display_rotation
        best: tuple[float, int, int, np.ndarray, np.ndarray] | None = None
        for geom_id, mesh in self.collision_meshes.items():
            geom_name = self.model.geom(geom_id).name or ""
            if elastomer_only and "elastomer" not in geom_name:
                continue
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            position = self.data.geom_xpos[geom_id]
            local_origin = (ray_origin - position) @ rotation
            local_direction = direction @ rotation
            locations, _, triangles = mesh.ray.intersects_location(
                [local_origin], [local_direction], multiple_hits=False
            )
            if not len(locations):
                continue
            point = locations[0] @ rotation.T + position
            distance = float(np.dot(point - ray_origin, direction))
            if distance < 0 or (best is not None and distance >= best[0]):
                continue
            normal = mesh.face_normals[triangles[0]] @ rotation.T
            best = (distance, geom_id, int(self.model.geom_bodyid[geom_id]), point, normal)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]


@dataclass
class Contact:
    serial: int
    body_id: int
    geom_name: str
    point_body: np.ndarray
    normal_body: np.ndarray
    frame: object


class TemplateEditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.shadow_keypoints = load_yaml(str(args.shadow_keypoints))
        for body_name, value in self.shadow_keypoints.items():
            if isinstance(value, str):
                self.shadow_keypoints[body_name] = self.shadow_keypoints[value]
        self.server = viser.ViserServer(host=args.host, port=args.port, label="Dexonomy")
        self.shadow = HandView(
            self.server, "shadow", args.shadow_xml, (-args.spacing / 2.0, 0.0, 0.0)
        )
        self.sharpa = HandView(
            self.server,
            "sharpa",
            args.sharpa_xml,
            (args.spacing / 2.0, 0.0, 0.0),
            display_rotation=SHARPA_DISPLAY_ROTATION,
        )
        if self.sharpa.model.nq != 22:
            raise ValueError(f"Expected 22 Sharpa DoF, found {self.sharpa.model.nq}")

        self.server.scene.add_label(
            "/shadow/title", "Shadow reference", position=(0.0, 0.0, 0.23), anchor="center-center"
        )
        self.server.scene.add_label(
            "/sharpa/title", "Sharpa left (editable)", position=(0.0, 0.0, 0.23), anchor="center-center"
        )
        self.contacts: list[Contact] = []
        self.contact_serial = 0
        self.shadow_contact_root = None
        self._updating_sliders = False
        self._build_gui()
        self._load_shadow_template(self.template_dropdown.value)

    def _build_gui(self) -> None:
        template_paths = sorted(self.args.shadow_templates.glob("*.yaml"))
        if not template_paths:
            raise FileNotFoundError(f"No Shadow templates in {self.args.shadow_templates}")
        self.template_by_name = {path.stem: path for path in template_paths}

        self.server.gui.add_markdown(
            "## Sharpa template editor\n左侧参考，右侧编辑。启用标注后点击 Sharpa 指腹。"
        )
        self.template_dropdown = self.server.gui.add_dropdown(
            "Shadow template", list(self.template_by_name), initial_value="14_Tripod"
            if "14_Tripod" in self.template_by_name
            else next(iter(self.template_by_name))
        )
        self.template_dropdown.on_update(
            lambda _: self._load_shadow_template(self.template_dropdown.value)
        )
        sharpa_paths = sorted(self.args.output_dir.glob("*.yaml"))
        self.sharpa_template_by_name = {path.stem: path for path in sharpa_paths}
        self.sharpa_template_dropdown = self.server.gui.add_dropdown(
            "Load Sharpa template", ["(new)", *self.sharpa_template_by_name]
        )
        self.sharpa_template_dropdown.on_update(
            lambda _: self._load_sharpa_template(self.sharpa_template_dropdown.value)
            if self.sharpa_template_dropdown.value != "(new)"
            else None
        )
        self.annotate_checkbox = self.server.gui.add_checkbox("Contact annotation mode", False)
        self.elastomer_checkbox = self.server.gui.add_checkbox("Elastomer only", True)
        collision_checkbox = self.server.gui.add_checkbox("Show Sharpa collision meshes", False)
        collision_checkbox.on_update(
            lambda _: [
                setattr(handle, "visible", collision_checkbox.value)
                for handle in self.sharpa.collision_handles.values()
            ]
        )
        self.template_name = self.server.gui.add_text("Output template name", "14_Tripod")
        self.overwrite_checkbox = self.server.gui.add_checkbox("Allow overwrite", False)
        self.status = self.server.gui.add_markdown("Ready")

        self.sliders = []
        prefixes = ("thumb", "index", "middle", "ring", "pinky")
        for prefix in prefixes:
            with self.server.gui.add_folder(prefix.capitalize(), expand_by_default=prefix == "thumb"):
                for joint_id in range(self.sharpa.model.njnt):
                    name = self.sharpa.model.joint(joint_id).name
                    if f"_{prefix}_" not in name:
                        continue
                    qpos_id = int(self.sharpa.model.jnt_qposadr[joint_id])
                    low, high = self.sharpa.model.jnt_range[joint_id]
                    slider = self.server.gui.add_slider(
                        name.removeprefix("left_"),
                        min=float(low),
                        max=float(high),
                        step=0.001,
                        initial_value=float(self.sharpa.data.qpos[qpos_id]),
                    )
                    slider.on_update(lambda _, q=qpos_id, s=slider: self._set_joint(q, s.value))
                    self.sliders.append((qpos_id, slider))

        with self.server.gui.add_folder("Actions", expand_by_default=True):
            self.normal_axis = self.server.gui.add_dropdown(
                "Last contact normal axis", ["X", "Y", "Z"], initial_value="X"
            )
            rotate_minus = self.server.gui.add_button("Rotate last normal -90°")
            rotate_minus.on_click(lambda _: self._rotate_last_normal(-90.0))
            rotate_plus = self.server.gui.add_button("Rotate last normal +90°")
            rotate_plus.on_click(lambda _: self._rotate_last_normal(90.0))
            reset = self.server.gui.add_button("Reset Sharpa pose")
            reset.on_click(lambda _: self._reset_pose())
            undo = self.server.gui.add_button("Undo last contact")
            undo.on_click(lambda _: self._undo_contact())
            clear = self.server.gui.add_button("Clear contacts")
            clear.on_click(lambda _: self._clear_contacts())
            save = self.server.gui.add_button("Save raw annotation", color="green")
            save.on_click(lambda _: self._save())

        self.server.scene.on_click()(self._on_scene_click)

    def _load_shadow_template(self, name: str) -> None:
        annotation = load_yaml(str(self.template_by_name[name]))
        self.template_name.value = name
        self.shadow.set_qpos(np.asarray(annotation["qpos"], dtype=float))
        if self.shadow_contact_root is not None:
            self.shadow_contact_root.remove()
        self.shadow_contact_root = self.server.scene.add_frame(
            "/shadow/reference_contacts", show_axes=False
        )
        for index, (body_name, raw_contact) in enumerate((annotation.get("contact") or {}).items()):
            for sub_index, value in enumerate(_contact_entries(raw_contact)):
                if isinstance(value, list):
                    point = np.asarray(value[:3], dtype=float)
                    normal = np.asarray(value[3:], dtype=float)
                else:
                    body_id = self.shadow.model.body(body_name).id
                    body_position, body_rotation = self.shadow.body_pose(body_id)
                    keypoint = np.asarray(
                        self.shadow_keypoints[body_name][int(value)], dtype=float
                    )
                    point = keypoint[:3] @ body_rotation.T + body_position
                    normal = keypoint[3:] @ body_rotation.T
                path = f"/shadow/reference_contacts/{index}_{sub_index}"
                self.server.scene.add_icosphere(
                    f"{path}/point", radius=0.003, color=(255, 80, 80), position=point
                )
                self.server.scene.add_arrows(
                    f"{path}/normal",
                    points=np.asarray([[point, point + 0.025 * normal]]),
                    colors=(255, 80, 80),
                    shaft_radius=0.001,
                    head_radius=0.003,
                    head_length=0.006,
                )
                self.server.scene.add_label(
                    f"{path}/label", body_name, position=point, font_screen_scale=0.7
                )
        self.status.content = f"Loaded Shadow template `{name}`"

    def _load_sharpa_template(self, name: str) -> None:
        annotation = load_yaml(str(self.sharpa_template_by_name[name]))
        qpos = np.asarray(annotation["qpos"], dtype=float)
        if qpos.shape != (self.sharpa.model.nq,):
            self.status.content = f"`{name}` has {len(qpos)} qpos values, expected 22"
            return
        self._set_sharpa_qpos(qpos)
        self._clear_contacts()
        for body_name, raw_contact in (annotation.get("contact") or {}).items():
            values = raw_contact if raw_contact and isinstance(raw_contact[0], list) else [raw_contact]
            body_id = self.sharpa.model.body(body_name).id
            body_position, body_rotation = self.sharpa.body_pose(body_id)
            for value in values:
                point = np.asarray(value[:3], dtype=float)
                normal = np.asarray(value[3:], dtype=float)
                point_body = (point - body_position) @ body_rotation
                normal_body = normal @ body_rotation
                geom_name = self._nearest_collision_geom(body_id, point)
                self._add_contact(body_id, geom_name, point_body, normal_body)
        self.template_name.value = name
        self.status.content = f"Loaded Sharpa template `{name}` ({len(self.contacts)} contacts)"

    def _set_sharpa_qpos(self, qpos: np.ndarray) -> None:
        self._updating_sliders = True
        self.sharpa.data.qpos[:] = qpos
        for qpos_id, slider in self.sliders:
            slider.value = float(qpos[qpos_id])
        self._updating_sliders = False
        mujoco.mj_forward(self.sharpa.model, self.sharpa.data)
        self.sharpa.update_visuals()
        self._update_contact_frames()

    def _nearest_collision_geom(self, body_id: int, point: np.ndarray) -> str:
        nearest = (float("inf"), "loaded contact")
        for geom_id, mesh in self.sharpa.collision_meshes.items():
            if int(self.sharpa.model.geom_bodyid[geom_id]) != body_id:
                continue
            rotation = self.sharpa.data.geom_xmat[geom_id].reshape(3, 3)
            local_point = (point - self.sharpa.data.geom_xpos[geom_id]) @ rotation
            _, distance, _ = mesh.nearest.on_surface([local_point])
            if distance[0] < nearest[0]:
                nearest = (
                    float(distance[0]),
                    self.sharpa.model.geom(geom_id).name or f"geom_{geom_id}",
                )
        return nearest[1]

    def _set_joint(self, qpos_id: int, value: float) -> None:
        if self._updating_sliders:
            return
        self.sharpa.data.qpos[qpos_id] = value
        mujoco.mj_forward(self.sharpa.model, self.sharpa.data)
        self.sharpa.update_visuals()
        self._update_contact_frames()

    def _reset_pose(self) -> None:
        self._set_sharpa_qpos(self.sharpa.model.qpos0)
        self.status.content = "Sharpa pose reset"

    def _update_contact_frames(self) -> None:
        for contact in self.contacts:
            position, rotation = self.sharpa.body_pose(contact.body_id)
            contact.frame.position = position
            contact.frame.wxyz = mat2quat(rotation)

    def _on_scene_click(self, event: viser.SceneClickEvent) -> None:
        if not self.annotate_checkbox.value:
            return
        hit = self.sharpa.raycast(
            np.asarray(event.ray_origin),
            np.asarray(event.ray_direction),
            self.elastomer_checkbox.value,
        )
        if hit is None:
            self.status.content = "No Sharpa collision surface hit"
            return
        geom_id, body_id, point, normal = hit
        body_position, body_rotation = self.sharpa.body_pose(body_id)
        point_body = (point - body_position) @ body_rotation
        normal_body = normal @ body_rotation
        geom_name = self.sharpa.model.geom(geom_id).name or f"geom_{geom_id}"
        self._add_contact(body_id, geom_name, point_body, normal_body)
        body_name = self.sharpa.model.body(body_id).name
        self.status.content = f"Added `{body_name}` on `{geom_name}` ({len(self.contacts)} total)"

    def _add_contact(
        self,
        body_id: int,
        geom_name: str,
        point_body: np.ndarray,
        normal_body: np.ndarray,
    ) -> None:
        serial = self.contact_serial
        self.contact_serial += 1
        contact = Contact(serial, body_id, geom_name, point_body, normal_body, None)
        contact.frame = self._draw_contact(contact)
        self.contacts.append(contact)

    def _draw_contact(self, contact: Contact) -> object:
        body_position, body_rotation = self.sharpa.body_pose(contact.body_id)
        serial = contact.serial
        frame = self.server.scene.add_frame(
            f"/sharpa/contacts/{serial}",
            show_axes=False,
            position=body_position,
            wxyz=mat2quat(body_rotation),
        )
        self.server.scene.add_icosphere(
            f"/sharpa/contacts/{serial}/point",
            radius=0.003,
            color=(255, 140, 0),
            position=contact.point_body,
        )
        self.server.scene.add_arrows(
            f"/sharpa/contacts/{serial}/normal",
            points=np.asarray(
                [[contact.point_body, contact.point_body + 0.025 * contact.normal_body]]
            ),
            colors=(255, 140, 0),
            shaft_radius=0.001,
            head_radius=0.003,
            head_length=0.006,
        )
        body_name = self.sharpa.model.body(contact.body_id).name
        self.server.scene.add_label(
            f"/sharpa/contacts/{serial}/label",
            f"{body_name} / {contact.geom_name}",
            position=contact.point_body,
            font_screen_scale=0.7,
        )
        return frame

    def _rotate_last_normal(self, angle_deg: float) -> None:
        if not self.contacts:
            self.status.content = "No contact normal to rotate"
            return
        contact = self.contacts[-1]
        axis = np.zeros(3)
        axis["XYZ".index(self.normal_axis.value)] = 1.0
        angle = np.deg2rad(angle_deg)
        normal = contact.normal_body
        contact.normal_body = (
            normal * np.cos(angle)
            + np.cross(axis, normal) * np.sin(angle)
            + axis * np.dot(axis, normal) * (1.0 - np.cos(angle))
        )
        contact.normal_body /= np.linalg.norm(contact.normal_body)
        contact.frame.remove()
        contact.frame = self._draw_contact(contact)
        body_name = self.sharpa.model.body(contact.body_id).name
        self.status.content = (
            f"Rotated last normal on `{body_name}` {angle_deg:+.0f}° "
            f"around local {self.normal_axis.value}"
        )

    def _undo_contact(self) -> None:
        if not self.contacts:
            self.status.content = "No contacts to undo"
            return
        self.contacts.pop().frame.remove()
        self.status.content = f"Removed last contact ({len(self.contacts)} remaining)"

    def _clear_contacts(self) -> None:
        while self.contacts:
            self.contacts.pop().frame.remove()
        self.status.content = "Contacts cleared"

    def _save(self) -> None:
        name = self.template_name.value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            self.status.content = "Template name may contain only letters, digits, `_`, and `-`"
            return
        output_path = self.args.output_dir / f"{name}.yaml"
        if output_path.exists() and not self.overwrite_checkbox.value:
            self.status.content = f"Refusing to overwrite existing `{output_path.name}`"
            return

        grouped: dict[str, list[list[float]]] = {}
        for contact in self.contacts:
            body_position, body_rotation = self.sharpa.body_pose(contact.body_id)
            point = contact.point_body @ body_rotation.T + body_position
            normal = contact.normal_body @ body_rotation.T
            body_name = self.sharpa.model.body(contact.body_id).name
            grouped.setdefault(body_name, []).append(
                np.round(np.concatenate([point, normal]), 6).tolist()
            )
        contact_yaml = {
            body: values[0] if len(values) == 1 else values for body, values in grouped.items()
        }
        payload = {
            "qpos": np.round(self.sharpa.data.qpos, 6).tolist(),
            "contact": contact_yaml,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        yaml = YAML()
        yaml.default_flow_style = None
        with output_path.open("w") as stream:
            yaml.dump(payload, stream)
        self.sharpa_template_by_name[name] = output_path
        self.sharpa_template_dropdown.options = ["(new)", *sorted(self.sharpa_template_by_name)]
        self.status.content = f"Saved `{output_path.relative_to(ROOT)}`"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8045)
    parser.add_argument("--spacing", type=float, default=0.28)
    parser.add_argument("--shadow-xml", type=Path, default=DEFAULT_SHADOW_XML)
    parser.add_argument("--shadow-templates", type=Path, default=DEFAULT_SHADOW_TEMPLATES)
    parser.add_argument("--shadow-keypoints", type=Path, default=DEFAULT_SHADOW_KEYPOINTS)
    parser.add_argument("--sharpa-xml", type=Path, default=DEFAULT_SHARPA_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Load assets and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.shadow_xml, args.shadow_templates, args.shadow_keypoints, args.sharpa_xml):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.check:
        shadow = mujoco.MjModel.from_xml_path(str(args.shadow_xml))
        sharpa = mujoco.MjModel.from_xml_path(str(args.sharpa_xml))
        templates = list(args.shadow_templates.glob("*.yaml"))
        assert shadow.nq == sharpa.nq == 22 and templates
        keypoints = load_yaml(str(args.shadow_keypoints))
        for body_name, value in keypoints.items():
            if isinstance(value, str):
                keypoints[body_name] = keypoints[value]
        for path in templates:
            annotation = load_yaml(str(path))
            assert len(annotation["qpos"]) == shadow.nq
            for body_name, raw_contact in (annotation.get("contact") or {}).items():
                for value in _contact_entries(raw_contact):
                    if not isinstance(value, list):
                        assert len(keypoints[body_name][int(value)]) == 6
        elastomer_id = next(
            i for i in range(sharpa.ngeom) if "elastomer" in (sharpa.geom(i).name or "")
        )
        mesh = _geom_mesh(sharpa, elastomer_id)
        point, normal = mesh.triangles_center[0], mesh.face_normals[0]
        hits, _, _ = mesh.ray.intersects_location(
            [point + 0.02 * normal], [-normal], multiple_hits=False
        )
        assert len(hits) == 1
        model_point = np.asarray([1.0, 2.0, 3.0])
        display_point = model_point @ SHARPA_DISPLAY_ROTATION.T
        assert np.allclose(display_point @ SHARPA_DISPLAY_ROTATION, model_point)
        sharpa_templates = list(args.output_dir.glob("*.yaml"))
        for path in sharpa_templates:
            annotation = load_yaml(str(path))
            assert len(annotation["qpos"]) == sharpa.nq
            for body_name, contacts in (annotation.get("contact") or {}).items():
                assert sharpa.body(body_name).id >= 0
                values = contacts if contacts and isinstance(contacts[0], list) else [contacts]
                assert all(len(value) == 6 for value in values)
        print(
            f"OK: Shadow nq={shadow.nq}, Sharpa nq={sharpa.nq}, "
            f"references={len(templates)}, Sharpa templates={len(sharpa_templates)}"
        )
        return
    TemplateEditor(args)
    print(f"Open http://{args.host}:{args.port}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
